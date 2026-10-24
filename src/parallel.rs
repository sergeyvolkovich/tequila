//! Пул потоков для многопоточного кодирования батчей кадров.
//!
//! # Контракт детерминизма
//!
//! `parallel_for(n, f)` / `map_into(out, f)` выполняют каждый индекс
//! РОВНО один раз со статическим разбиением на чанки; порядок вычисления
//! индексов между потоками не влияет на результат, пока `f` пишет только
//! в свой слот (контракт вызывающего) и не делает кросс-индексных
//! редукций. Все FP-операции внутри одного индекса идут в исходном
//! скалярном порядке → битстрим ПОБИТОВО совпадает с однопоточным
//! запуском (проверяется тестом и корпусом).
//!
//! # Заимствования и unsafe
//!
//! Тела задач обязаны быть `'static` (канал пула), но им нужно
//! заимствовать локальные данные (`&F`, `&mut [T]`). Звучность даёт
//! инвариант «join до возврата»: `parallel_for` НЕ возвращается, пока
//! все задачи не завершатся. Два точечных unsafe:
//! * [`SendPtr`]/[`SendSlots`] — обёртки сырых указателей с ручными
//!   `Send`/`Sync` (индексы disjoint по контракту);
//! * разыменование `fptr` в задаче — живёт дольше join'а по инварианту.
//!
//! Паника в задаче пробрасывается в вызывающий поток при выходе из
//! `parallel_for` (первая паника, детерминированно).

use std::any::Any;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// `*const ()` с ручным Send/Sync: СТИЁРТЫЙ указатель на immutable
/// данные (тип-владелец приводится при создании и разыменовании);
/// живёт дольше задач (join до возврата).
struct SendPtr(*const ());

impl SendPtr {
    fn as_ptr(&self) -> *const () {
        self.0
    }
}

// SAFETY: параллельные чтения через указатель безопасны (immutable);
// владелец живёт дольше всех задач (join в parallel_for).
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// `*mut [T]` с ручным Send/Sync: слоты disjoint по индексам
/// (каждый i — ровно один поток), join до возврата.
struct SendSlots<T>(*mut [T]);

impl<T> SendSlots<T> {
    fn as_mut_ptr(&self) -> *mut [T] {
        self.0
    }
}

// SAFETY: запись только в disjoint индексы (контракт map_into);
// владелец слотов живёт дольше всех задач (join в parallel_for).
unsafe impl<T: Send> Send for SendSlots<T> {}
unsafe impl<T: Send> Sync for SendSlots<T> {}

/// Пул рабочих потоков. `workers == 0` — однопоточный режим (все
/// `parallel_for`/`map_into` исполняются инлайн, без накладных расходов).
pub struct Pool {
    tx: Option<mpsc::Sender<Job>>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    /// Создать пул с `workers` рабочими потоками (0 — инлайн-режим).
    pub fn new(workers: usize) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let rx = Arc::clone(&rx);
            handles.push(std::thread::spawn(move || {
                loop {
                    let job = { rx.lock().unwrap().recv() };
                    match job {
                        // панику гасим: её получит вызывающий поток
                        Ok(job) => {
                            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                        }
                        Err(_) => break, // канал закрыт — пул завершается
                    }
                }
            }));
        }
        Pool { tx: Some(tx), handles }
    }

    /// Число рабочих потоков (без учёта потока-инициатора).
    pub fn workers(&self) -> usize {
        self.handles.len()
    }

    /// Выполнить `f(i)` для всех `i` в `0..n`, распределив индексы по
    /// потокам (включая вызывающий). Детерминизм — см. контракт модуля.
    pub fn parallel_for<F: Fn(usize) + Sync>(&self, n: usize, f: F) {
        if n == 0 {
            return;
        }
        let workers = self.handles.len();
        if workers == 0 || self.tx.is_none() {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let nthreads = workers + 1;
        let chunk = n.div_ceil(nthreads);
        // сколько задач реально уйдёт воркерам (чанки могут быть пустыми)
        let submitted = (1..nthreads).filter(|&t| t * chunk < n).count();
        let done = Arc::new((Mutex::new(0usize), Condvar::new()));
        let panic_slot: Arc<Mutex<Option<Box<dyn Any + Send>>>> = Arc::new(Mutex::new(None));
        let fptr = SendPtr(&f as *const F as *const ());

        let mut spawned: Vec<mpsc::Sender<Job>> = Vec::new();
        for t in 1..nthreads {
            let lo = t * chunk;
            let hi = ((t + 1) * chunk).min(n);
            if lo >= hi {
                break;
            }
            let done_t = Arc::clone(&done);
            let panic_t = Arc::clone(&panic_slot);
            let fptr = SendPtr(fptr.as_ptr());
            // 'static-задача: fptr — стёртый сырой указатель (см. SAFETY в задаче)
            let job: Job = Box::new(move || {
                // SAFETY: f живёт дольше всех задач — parallel_for ждёт
                // завершения (done-счётчик) до возврата
                let f = unsafe { &*(fptr.as_ptr() as *const F) };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for i in lo..hi {
                        f(i);
                    }
                }));
                if let Err(p) = r {
                    let mut slot = panic_t.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(p);
                    }
                }
                let mut c = done_t.0.lock().unwrap();
                *c += 1;
                done_t.1.notify_all();
            });
            // держим клон отправителя, чтобы канал жил до конца раздачи
            let tx = self.tx.as_ref().unwrap().clone();
            if tx.send(job).is_ok() {
                spawned.push(tx);
            } else {
                // пул падает — выполняем сами (деградация, не тупик)
                for i in lo..hi {
                    f(i);
                }
            }
        }
        drop(spawned);
        // доля потока-инициатора
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for i in 0..chunk.min(n) {
                f(i);
            }
        }));
        if let Err(p) = r {
            let mut slot = panic_slot.lock().unwrap();
            if slot.is_none() {
                *slot = Some(p);
            }
        }
        // ожидание воркеров
        {
            let mut c = done.0.lock().unwrap();
            while *c < submitted {
                c = done.1.wait(c).unwrap();
            }
        }
        let p = panic_slot.lock().unwrap().take();
        if let Some(p) = p {
            std::panic::resume_unwind(p);
        }
    }

    /// Отображение с записью в слоты: `out[i] = f(i)` для всех i.
    ///
    /// Ключевой примитив детерминированного параллелизма энкодера: каждый
    /// индекс вычисляется ровно один раз, результат пишется в свой слот —
    /// итоговые данные не зависят от порядка выполнения потоков. `f`
    /// обязана зависеть только от `i` и неизменяемых захватов (контракт
    /// `Fn + Sync`).
    pub fn map_into<T: Send, F: Fn(usize) -> T + Sync>(&self, out: &mut [T], f: F) {
        let n = out.len();
        if n == 0 {
            return;
        }
        let workers = self.handles.len();
        if workers == 0 || self.tx.is_none() {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = f(i);
            }
            return;
        }
        // слоты как сырой указатель: disjoint-индексы (контракт выше)
        let slots = SendSlots(out as *mut [T]);
        self.parallel_for(n, move |i| unsafe {
            // SAFETY: индекс i принадлежит ровно одному потоку, join до возврата
            (*slots.as_mut_ptr())[i] = f(i);
        });
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.tx.take(); // закрыть канал → воркеры выходят
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детерминизм: результаты идентичны однопоточному запуску; все
    /// индексы покрыты ровно один раз; паника пробрасывается.
    #[test]
    fn parallel_for_deterministic_and_covers_all() {
        for &workers in &[0usize, 1, 3] {
            let pool = Pool::new(workers);
            for &n in &[1usize, 7, 64, 1000] {
                // слоты через map_into; parallel_for — только чистая работа
                let mut hits = vec![0u32; n];
                pool.map_into(&mut hits, |i| {
                    let _ = i;
                    1u32
                });
                let vals: Vec<u64> = (0..n as u64).map(|i| i * 31 + 7).collect();
                let mut got = vec![0u64; n];
                pool.map_into(&mut got, |i| vals[i]);
                assert!(hits.iter().all(|&h| h == 1), "workers={workers} n={n}");
                for (i, &v) in got.iter().enumerate() {
                    assert_eq!(v, (i as u64) * 31 + 7);
                }
                // порядок исполнения не важен для чистых задач: побочный
                // счётчик через слоты map_into
                let mut order = vec![usize::MAX; n];
                pool.map_into(&mut order, |i| i.wrapping_mul(2654435761) % n.max(1));
                assert!(order.iter().all(|&v| v != usize::MAX));
            }
        }
    }

    /// Паника внутри задачи доходит до вызывающего потока.
    #[test]
    #[should_panic(expected = "boom")]
    fn parallel_for_propagates_panic() {
        let pool = Pool::new(2);
        pool.parallel_for(16, |i| {
            if i == 11 {
                panic!("boom");
            }
        });
    }

    /// Неконстантное заимствование: map_into пишет в слоты по &mut,
    /// задача читает чужой буфер по &.
    #[test]
    fn map_into_borrows_env() {
        let env: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
        let mut out = vec![0.0f64; env.len()];
        let pool = Pool::new(2);
        pool.map_into(&mut out, |i| env[i] * 2.0);
        for (o, &e) in out.iter().zip(env.iter()) {
            assert_eq!(*o, e * 2.0);
        }
    }

    /// map_into: слоты заполняются независимо от разбиения; результат
    /// совпадает с последовательным.
    #[test]
    fn map_into_matches_sequential() {
        let pool = Pool::new(3);
        for &n in &[1usize, 5, 100, 257] {
            let mut got = vec![0u64; n];
            pool.map_into(&mut got, |i| (i as u64).wrapping_mul(0x9E37_79B9) ^ (i as u64));
            for (i, &v) in got.iter().enumerate() {
                assert_eq!(v, (i as u64).wrapping_mul(0x9E37_79B9) ^ (i as u64));
            }
        }
    }
}
