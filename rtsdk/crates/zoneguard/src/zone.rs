//! ゾーンと、ゾーンごとに 1 枚しか存在しない排他トークン。
//!
//! ゾーンは「型」である。`ZoneX` に入る許可 `Permit<ZoneX>` はプロセス全体で
//! **ちょうど 1 個**しか作られず、`Clone` も `Copy` も出来ない。
//! したがって「2 つの機構が同時にゾーン X の中に居る」状態は、
//! 実行時のフラグではなく所有権の規則で不可能になる。

use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::geom::{Aabb, P3};

/// 排他ゾーンの定義。座標も ID もすべてコンパイル時に確定する。
pub trait Zone: 'static {
    const ID: u16;
    const NAME: &'static str;
    const BOX: Aabb;
}

/// 2 本のアームが取り合う共有ワークエリア。
pub struct ZoneX;

impl Zone for ZoneX {
    const ID: u16 = 1;
    const NAME: &'static str = "ZoneX";
    /// x = 150〜250 mm の帯。2 本のアームはここへ左右から入ってくる。
    const BOX: Aabb = Aabb::centered(P3::new(200_000, 0, 0), P3::new(50_000, 60_000, 60_000));
}

/// 段取り替え用の別ゾーン。`ZoneX` とは型が違うので取り合いにならない —
/// 「別のゾーンなら並行してよい」が型の上でそのまま出る。
pub struct ZoneY;

impl Zone for ZoneY {
    const ID: u16 = 2;
    const BOX: Aabb = Aabb::centered(P3::new(600_000, 0, 0), P3::new(40_000, 60_000, 60_000));
    const NAME: &'static str = "ZoneY";
}

/// ゾーン `Z` へ入る許可。複製できず、移動しかできない (線形資源として扱う)。
///
/// これを持っている間だけ `Z` に触れる。誰かに渡したら自分はもう触れない。
pub struct Permit<Z: Zone> {
    _z: PhantomData<fn() -> Z>,
}

impl<Z: Zone> Permit<Z> {
    /// クレート内部だけが発行できる。外から増やす手段は無い。
    pub(crate) const fn mint() -> Self {
        Permit { _z: PhantomData }
    }

    pub const fn zone_id(&self) -> u16 {
        Z::ID
    }

    pub const fn zone_box(&self) -> Aabb {
        Z::BOX
    }
}

impl<Z: Zone> core::fmt::Debug for Permit<Z> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Permit<{}>", Z::NAME)
    }
}

static INTERLOCK_TAKEN: AtomicBool = AtomicBool::new(false);

/// 設備の起動時に 1 度だけ組み立てる、全ゾーンの許可の束。
///
/// **この構造体を作れるのは 1 回だけ** — 唯一の実行時チェックがここにあり、
/// 以降の排他はすべてコンパイル時に解決される。
pub struct Interlock {
    pub zone_x: Permit<ZoneX>,
    pub zone_y: Permit<ZoneY>,
}

impl Interlock {
    /// 2 度目以降は `None`。「許可証を刷り増す」経路が存在しないことを保証する。
    pub fn take() -> Option<Interlock> {
        if INTERLOCK_TAKEN.swap(true, Ordering::SeqCst) {
            None
        } else {
            Some(Interlock {
                zone_x: Permit::mint(),
                zone_y: Permit::mint(),
            })
        }
    }

    /// テストハーネス専用。シナリオの間で設備を組み直すために巻き戻す。
    ///
    /// メモリ安全性には触れない (トークンは ZST) が、**プロトコルの唯一性を壊す**ので、
    /// 呼んでよいのは 1 シナリオを直列に囲っているハーネスだけ。
    #[doc(hidden)]
    pub fn rearm_for_harness() {
        INTERLOCK_TAKEN.store(false, Ordering::SeqCst);
    }
}

/// ゾーン `Z` が空いていることを要求する操作の例。
///
/// 引数が `&Permit<Z>` なので、**許可証が誰かの型状態の中に入っている間は呼べない**。
/// アームがゾーン X に入っている間、コンベアは動かせない — 実行時の判定ではなく、
/// 借りられる参照が存在しないという理由で。
///
/// ```
/// use zoneguard::{Interlock, conveyor_advance_while_free};
/// let il = Interlock::take().unwrap();
/// // 誰もゾーン X に居ないので、許可証を借りられる = コンベアを回せる。
/// assert_eq!(conveyor_advance_while_free(&il.zone_x, 10_000), 10_000);
/// # Interlock::rearm_for_harness();
/// ```
pub fn conveyor_advance_while_free<Z: Zone>(_free: &Permit<Z>, pitch_um: i32) -> i32 {
    pitch_um
}
