//! 仮想時間モナド。
//!
//! この層の目的は 3 つだけ。
//!
//! 1. **実時間のシステムコールを消す。** `usleep` / `clock_gettime` はここに無い。
//!    時計は [`Ctx`] が持つただの `u64` で、進めるのは [`Delay`] か [`Compute`] だけ。
//!    Δt (制御周期の刻み) と開始時刻は外 — シミュレータや Wasm ホスト — から注入する。
//! 2. **合成した手続きの WCET を型の上で足す。** [`Prog::WCET_US`] は関連定数なので、
//!    `bind` で繋いだ瞬間に和がコンパイル時に決まる。周期予算との比較は
//!    [`Budget`] で静的アサーションになる (超えていればビルドが通らない)。
//! 3. **動的確保をしない。** `no_std` で `alloc` を持ち込まない。合成は
//!    すべて単相化される構造体で、`dyn` も `Box` も経由しない。
//!
//! ```
//! use vtime::{Ctx, Prog, compute, delay_us, pure, Budget};
//!
//! // 各ステップは「自分の最悪計算時間」を型引数として申告する。
//! fn step() -> impl Prog<Out = u32> {
//!     compute::<30, _, _>(|| 7u32).bind(|v| delay_us(200).map(move |_| v * 2))
//! }
//!
//! let mut ctx = Ctx::new(0, 100, ());   // 開始 0 µs、Δt = 100 µs
//! let v = step().run(&mut ctx);
//! assert_eq!(v, 14);
//! assert_eq!(ctx.cpu_us(), 30);         // 申告した計算時間
//! assert_eq!(ctx.now(), 300);           // 30 µs は Δt=100 に丸められ、+200 µs
//! ```
#![no_std]
#![forbid(unsafe_code)]

use core::marker::PhantomData;
use pulse_trace::{Ev, Kind, Sink};

pub use pulse_trace::{Ev as TraceEv, Kind as TraceKind, Sink as TraceSink};

/// マイクロ秒。この SDK に浮動小数点の時刻は無い。
pub type Micros = u64;

/// 仮想時間の文脈。時計と、申告済み計算時間の累計と、イベントの出口を持つ。
///
/// `S` は単相化されるので、記録の有無で WCET が変わらないことを型で担保できる
/// (`()` を渡せば記録は消える)。
pub struct Ctx<S: Sink> {
    now: Micros,
    dt_us: Micros,
    cpu_us: Micros,
    sink: S,
}

impl<S: Sink> Ctx<S> {
    /// 開始時刻と Δt を外から注入する。`dt_us = 0` は「刻み無し (連続)」の意味。
    pub const fn new(start_us: Micros, dt_us: Micros, sink: S) -> Self {
        Ctx {
            now: start_us,
            dt_us,
            cpu_us: 0,
            sink,
        }
    }

    pub const fn now(&self) -> Micros {
        self.now
    }

    pub const fn dt_us(&self) -> Micros {
        self.dt_us
    }

    /// 申告 WCET の累計。実測ではなく「最悪ケースで回した」値。
    pub const fn cpu_us(&self) -> Micros {
        self.cpu_us
    }

    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    pub fn into_sink(self) -> S {
        self.sink
    }

    pub fn emit(&mut self, lane: u16, kind: Kind, a: i32, b: i32) {
        let t = self.now;
        self.sink.emit(Ev::new(t, lane, kind, a, b));
    }

    /// 時計を進める。Δt が入っていれば、その刻みの次の境界まで切り上げる —
    /// 周期実行系では「途中の時刻」は観測されないため。
    pub fn advance(&mut self, us: Micros) {
        let t = self.now + us;
        self.now = quantize(t, self.dt_us);
    }

    /// 申告した最悪計算時間を計上する。SDK の各操作 (`zoneguard` のアーム指令など) が
    /// 自分の `WCET_US` を持ち込むための口。
    pub fn charge_us(&mut self, wcet_us: Micros) {
        self.cpu_us += wcet_us;
        self.advance(wcet_us);
    }

    fn charge(&mut self, wcet_us: Micros) {
        self.charge_us(wcet_us)
    }
}

/// `dt` の刻みに切り上げる。`dt = 0` なら素通し。
pub const fn quantize(t: Micros, dt: Micros) -> Micros {
    if dt == 0 {
        t
    } else {
        t.div_ceil(dt) * dt
    }
}

/// 仮想時間の上で走る手続き。
///
/// `run` を呼ぶまで何も起きない (記述と実行が分かれている) ので、
/// 「この手続きの WCET はいくつか」を実行せずに型から取り出せる。
pub trait Prog: Sized {
    type Out;

    /// 静的に見積もった最悪計算時間 [µs]。合成すると和になる。
    const WCET_US: Micros;

    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> Self::Out;

    /// モナドの `>>=`。次の手続きは直前の結果から決まる。
    fn bind<N, F>(self, f: F) -> Bind<Self, F, N>
    where
        N: Prog,
        F: FnOnce(Self::Out) -> N,
    {
        Bind {
            m: self,
            f,
            _n: PhantomData,
        }
    }

    fn map<B, F>(self, f: F) -> Map<Self, F, B>
    where
        F: FnOnce(Self::Out) -> B,
    {
        Map {
            m: self,
            f,
            _b: PhantomData,
        }
    }

    /// 結果を捨てて次へ。`bind(|_| n)` と同じだが型が読みやすい。
    fn then<N: Prog>(self, n: N) -> Then<Self, N> {
        Then { m: self, n }
    }
}

/// 何もせず値を返す。WCET 0。
pub struct Pure<A>(pub A);

pub const fn pure<A>(a: A) -> Pure<A> {
    Pure(a)
}

impl<A> Prog for Pure<A> {
    type Out = A;
    const WCET_US: Micros = 0;
    fn run<S: Sink>(self, _ctx: &mut Ctx<S>) -> A {
        self.0
    }
}

/// 申告 WCET つきの純計算。`usleep` ではなく「この計算に最大 W µs かかる」の宣言。
pub struct Compute<F, B, const W: Micros> {
    f: F,
    _b: PhantomData<fn() -> B>,
}

/// `compute::<128, _, _>(|| ...)` — 128 µs を最悪計算時間として申告する。
pub const fn compute<const W: Micros, B, F: FnOnce() -> B>(f: F) -> Compute<F, B, W> {
    Compute { f, _b: PhantomData }
}

impl<F: FnOnce() -> B, B, const W: Micros> Prog for Compute<F, B, W> {
    type Out = B;
    const WCET_US: Micros = W;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> B {
        let out = (self.f)();
        ctx.charge(W);
        out
    }
}

/// 待ち。`usleep` の置き換えで、CPU は使わず時計だけが進む。
pub struct Delay(pub Micros);

pub const fn delay_us(us: Micros) -> Delay {
    Delay(us)
}

impl Prog for Delay {
    type Out = ();
    const WCET_US: Micros = 0;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) {
        ctx.advance(self.0);
    }
}

/// トレースへ 1 イベント落とす。
pub struct Emit {
    pub lane: u16,
    pub kind: Kind,
    pub a: i32,
    pub b: i32,
}

pub const fn emit(lane: u16, kind: Kind, a: i32, b: i32) -> Emit {
    Emit { lane, kind, a, b }
}

impl Prog for Emit {
    type Out = ();
    /// 記録そのものにも時間はかかる。0 にしてしまうと計測が嘘になる。
    const WCET_US: Micros = 1;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) {
        ctx.emit(self.lane, self.kind, self.a, self.b);
        ctx.charge(Self::WCET_US);
    }
}

/// 現在の仮想時刻を読む。
pub struct Now;

impl Prog for Now {
    type Out = Micros;
    const WCET_US: Micros = 0;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> Micros {
        ctx.now()
    }
}

pub struct Bind<M, F, N> {
    m: M,
    f: F,
    _n: PhantomData<fn() -> N>,
}

impl<M, F, N> Prog for Bind<M, F, N>
where
    M: Prog,
    N: Prog,
    F: FnOnce(M::Out) -> N,
{
    type Out = N::Out;
    /// ここが肝。合成した瞬間に WCET が型の上で足される。
    const WCET_US: Micros = M::WCET_US + N::WCET_US;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> N::Out {
        let a = self.m.run(ctx);
        (self.f)(a).run(ctx)
    }
}

pub struct Map<M, F, B> {
    m: M,
    f: F,
    _b: PhantomData<fn() -> B>,
}

impl<M: Prog, F: FnOnce(M::Out) -> B, B> Prog for Map<M, F, B> {
    type Out = B;
    const WCET_US: Micros = M::WCET_US;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> B {
        let a = self.m.run(ctx);
        (self.f)(a)
    }
}

pub struct Then<M, N> {
    m: M,
    n: N,
}

impl<M: Prog, N: Prog> Prog for Then<M, N> {
    type Out = N::Out;
    const WCET_US: Micros = M::WCET_US + N::WCET_US;
    fn run<S: Sink>(self, ctx: &mut Ctx<S>) -> N::Out {
        self.m.run(ctx);
        self.n.run(ctx)
    }
}

/// 周期予算に対する静的検査。
///
/// ```
/// use vtime::Budget;
/// type Step = vtime::Compute<fn() -> (), (), 400>;
/// const _: () = Budget::<Step, 1_000>::FITS;   // 400 µs ≤ 1 ms、通る
/// ```
///
/// 予算を超えていればビルドが落ちる。実機に載せる前に、机の上で落ちる。
///
/// ```compile_fail
/// use vtime::Budget;
/// type Step = vtime::Compute<fn() -> (), (), 400>;
/// const _: () = Budget::<Step, 100>::FITS;     // 400 µs > 100 µs
/// ```
pub struct Budget<P, const PERIOD_US: Micros>(PhantomData<fn() -> P>);

impl<P: Prog, const PERIOD_US: Micros> Budget<P, PERIOD_US> {
    /// 評価された時点で `P::WCET_US <= PERIOD_US` を要求する。
    pub const FITS: () = assert!(
        P::WCET_US <= PERIOD_US,
        "WCET が周期予算を超えている: この手続きはこの周期に載らない"
    );

    /// 余裕 [µs]。負にはならない (負なら `FITS` 側でビルドが落ちる)。
    pub const SLACK_US: Micros = PERIOD_US.saturating_sub(P::WCET_US);
}

/// do 記法。`let x = prog;` を並べると `bind` の連鎖に展開される。
///
/// ```
/// use vtime::{timed, Ctx, Prog, compute, delay_us, pure};
///
/// let prog = timed! {
///     let a = compute::<10, _, _>(|| 2u32);
///     let _ = delay_us(500);
///     let b = compute::<10, _, _>(move || a + 3);
///     pure(b)
/// };
/// let mut ctx = Ctx::new(0, 0, ());
/// assert_eq!(prog.run(&mut ctx), 5);
/// assert_eq!(ctx.now(), 520);
/// ```
#[macro_export]
macro_rules! timed {
    (let $x:pat = $p:expr; $($rest:tt)*) => {
        $crate::Prog::bind($p, move |$x| $crate::timed!($($rest)*))
    };
    ($p:expr $(;)?) => { $p };
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulse_trace::Recorder;

    fn cycle() -> impl Prog<Out = u32> {
        timed! {
            let _ = emit(0, Kind::TaskRise, 0, 0);
            let x = compute::<120, _, _>(|| 21u32);
            let _ = delay_us(1_000);
            let y = compute::<80, _, _>(move || x * 2);
            let _ = emit(0, Kind::TaskFall, 0, 0);
            pure(y)
        }
    }

    #[test]
    fn wcet_is_the_sum_of_the_parts_at_compile_time() {
        // 1 + 120 + 0 + 80 + 1 = 202。実行前に、型だけから取れる。
        fn wcet_of<P: Prog>(_p: &P) -> Micros {
            P::WCET_US
        }
        assert_eq!(wcet_of(&cycle()), 202);
    }

    #[test]
    fn delay_does_not_consume_cpu_budget() {
        let mut ctx = Ctx::new(0, 0, ());
        let out = cycle().run(&mut ctx);
        assert_eq!(out, 42);
        assert_eq!(ctx.cpu_us(), 202);
        assert_eq!(ctx.now(), 1_202);
    }

    #[test]
    fn dt_injection_snaps_to_the_control_period() {
        let mut ctx = Ctx::new(0, 250, ());
        cycle().run(&mut ctx);
        // すべての事象が 250 µs 格子の上に乗る。
        assert_eq!(ctx.now() % 250, 0);
    }

    #[test]
    fn the_sink_sees_the_edges_with_virtual_timestamps() {
        let mut rec = Recorder::<16>::new();
        let mut ctx = Ctx::new(1_000, 0, &mut rec);
        cycle().run(&mut ctx);
        let evs = rec.events();
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].kind, Kind::TaskRise);
        assert_eq!(evs[0].t_us, 1_000);
        assert_eq!(evs[1].kind, Kind::TaskFall);
        assert_eq!(evs[1].t_us, 1_000 + 202 - 1 + 1_000);
    }

    #[test]
    fn same_program_same_trace_regardless_of_wall_clock() {
        // 2 回走らせて完全に一致する = 実時間に依存する経路が無い。
        let mut a = Recorder::<16>::new();
        let mut b = Recorder::<16>::new();
        cycle().run(&mut Ctx::new(0, 100, &mut a));
        cycle().run(&mut Ctx::new(0, 100, &mut b));
        assert_eq!(a.events(), b.events());
    }
}
