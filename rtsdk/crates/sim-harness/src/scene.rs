//! 3D 空間のセマンティクスを宣言する DSL と、その型。
//!
//! 現場で図面に書いてある約束 —「この 2 つは 50 mm 以内に近づけてはいけない」
//! 「このエリアには 1 台しか入れない」「1 サイクルは 400 ms 以内」— を
//! そのままの語彙で書き、[`semantics!`] がそれを
//!
//! * 実行時に評価できる規則の表 ([`SceneDef`]) と、
//! * 規則 1 本ごとの `#[test]` ([`verify_scene!`])
//!
//! の両方へ落とす。図面と検証コードが 1 つの記述から出るので、ずれない。

pub use zoneguard::{Aabb, P3};

/// 干渉判定に参加する剛体 (アームの先端など)。
///
/// 位置は `base + dir * (軸の指令値)`。軸座標から世界座標への写像をここで持つ。
#[derive(Clone, Copy, Debug)]
pub struct BodyDef {
    pub name: &'static str,
    pub axis: u8,
    pub base: P3,
    /// 各成分が -1 / 0 / 1 の方向。
    pub dir: P3,
    /// 包絡球の半径 [µm]。干渉距離は表面間で測る。
    pub radius_um: i32,
}

/// 排他エリア。
#[derive(Clone, Copy, Debug)]
pub struct ZoneDef {
    pub name: &'static str,
    pub id: u16,
    pub aabb: Aabb,
}

/// 守るべき規則。配列上の位置がそのまま規則 id になる。
#[derive(Clone, Copy, Debug)]
pub enum RuleDef {
    /// 2 つの剛体の表面間距離が `min_um` を下回らないこと。
    Separation {
        name: &'static str,
        a: usize,
        b: usize,
        min_um: i32,
    },
    /// ゾーンに同時に 2 つ以上入らないこと。
    Exclusive { name: &'static str, zone: usize },
    /// 1 サイクルが `budget_us` 以内に終わること。
    Deadline { name: &'static str, budget_us: u64 },
}

impl RuleDef {
    pub const fn name(&self) -> &'static str {
        match self {
            RuleDef::Separation { name, .. } => name,
            RuleDef::Exclusive { name, .. } => name,
            RuleDef::Deadline { name, .. } => name,
        }
    }
}

pub struct SceneDef {
    pub name: &'static str,
    pub desc: &'static str,
    pub bodies: &'static [BodyDef],
    pub zones: &'static [ZoneDef],
    pub rules: &'static [RuleDef],
}

impl SceneDef {
    /// 剛体の世界座標。軸の指令値 `pos_um` から求める。
    pub fn tip(&self, body: usize, pos_um: i32) -> P3 {
        let b = &self.bodies[body];
        b.base.add(P3::along(b.dir, pos_um))
    }

    pub fn rule(&self, id: u16) -> &RuleDef {
        &self.rules[id as usize]
    }

    pub fn deadline_us(&self) -> Option<u64> {
        self.rules.iter().find_map(|r| match r {
            RuleDef::Deadline { budget_us, .. } => Some(*budget_us),
            _ => None,
        })
    }
}

/// 規則 1 本を組み立てる補助マクロ。[`semantics!`] から呼ばれる。
#[macro_export]
#[doc(hidden)]
macro_rules! rule_def {
    ($n:ident : separation($a:ident, $b:ident) >= $v:literal) => {
        $crate::scene::RuleDef::Separation {
            name: stringify!($n),
            a: $a,
            b: $b,
            min_um: $v,
        }
    };
    ($n:ident : exclusive($z:ident)) => {
        $crate::scene::RuleDef::Exclusive {
            name: stringify!($n),
            zone: $z,
        }
    };
    ($n:ident : deadline <= $v:literal) => {
        $crate::scene::RuleDef::Deadline {
            name: stringify!($n),
            budget_us: $v,
        }
    };
}

/// 3D 空間のセマンティクスを 1 か所で宣言する。
///
/// ```
/// use sim_harness::semantics;
///
/// semantics! {
///     scene cell01 "2 本のアームが 1 つのワークエリアを取り合う" {
///         bodies {
///             [ARM_A: axis 0, base (0, 0, 0), dir (1, 0, 0), radius 30_000]
///             [ARM_B: axis 1, base (400_000, 0, 0), dir (-1, 0, 0), radius 30_000]
///         }
///         zones {
///             [ZONE_X: id 1, center (200_000, 0, 0), half (50_000, 60_000, 60_000)]
///         }
///         rules {
///             [SEPARATION: separation(ARM_A, ARM_B) >= 50_000]
///             [EXCLUSIVE_X: exclusive(ZONE_X)]
///             [TACT: deadline <= 400_000]
///         }
///     }
/// }
///
/// assert_eq!(cell01::SCENE.rules.len(), 3);
/// assert_eq!(cell01::SEPARATION, 0);      // 規則 id は宣言順
/// assert_eq!(cell01::ARM_B, 1);           // 剛体 id も宣言順
/// ```
#[macro_export]
macro_rules! semantics {
    (
        scene $sname:ident $desc:literal {
            bodies $bodies:tt
            zones $zones:tt
            rules $rules:tt
        }
    ) => {
        $crate::semantics! {
            scene $sname $desc {
                bodies $bodies
                zones $zones
                rules $rules
                verify {}
            }
        }
    };

    (
        scene $sname:ident $desc:literal {
            bodies $bodies:tt
            zones $zones:tt
            rules $rules:tt
            verify $verify:tt
        }
    ) => {
        pub mod $sname {
            #![allow(dead_code, unused_imports)]
            use $crate::scene::{Aabb, BodyDef, RuleDef, SceneDef, ZoneDef, P3};

            $crate::semantics!(@body_ids 0usize; $bodies);
            $crate::semantics!(@zone_ids 0usize; $zones);
            $crate::semantics!(@rule_ids 0u16; $rules);

            pub static SCENE: SceneDef = SceneDef {
                name: stringify!($sname),
                desc: $desc,
                bodies: $crate::semantics!(@body_arr $bodies),
                zones: $crate::semantics!(@zone_arr $zones),
                rules: $crate::semantics!(@rule_arr $rules),
            };

            $crate::semantics!(@verify $rules; $verify);
        }
    };

    // --- 規則 1 本ごとに #[test] を生やす ---
    (@verify $rules:tt; { $($items:tt)* }) => {
        $crate::semantics!(@verify_m $rules; $($items)*);
    };
    (@verify_m $rules:tt;) => {};
    (@verify_m $rules:tt; [green $lab:ident = $p:path] $($tail:tt)*) => {
        /// 型安全な器に入れた走行。宣言した規則がすべて成り立つことを、規則ごとに見る。
        #[cfg(test)]
        #[allow(non_snake_case)]
        pub mod $lab {
            $crate::semantics!(@green_tests $p; $rules);
        }
        $crate::semantics!(@verify_m $rules; $($tail)*);
    };
    (@verify_m $rules:tt; [red $lab:ident = $p:path] $($tail:tt)*) => {
        /// 包む前の走行。個々の規則テストは `--ignored` を付けたときだけ走り、
        /// 走らせれば赤くなる (それが出発点なので、CI では既定で伏せてある)。
        /// 常時走るのは「まだ赤いままか」の 1 本だけ。
        #[cfg(test)]
        #[allow(non_snake_case)]
        pub mod $lab {
            $crate::semantics!(@red_tests $p; $rules);

            #[test]
            fn premise_is_still_red() {
                let r = $p(&super::SCENE);
                assert!(
                    !r.is_clean(),
                    "\n[{}] {} が違反ゼロで通ってしまった。\n  この走行は「型で包む前は破れる」ことを示すための前提。\n  レガシー側かシナリオが変わっている。TDD の出発点を確認しなおすこと。\n",
                    super::SCENE.name,
                    r.label
                );
            }
        }
        $crate::semantics!(@verify_m $rules; $($tail)*);
    };

    (@green_tests $p:path; { $([$rn:ident : $($rb:tt)+])+ }) => {
        $(
            #[test]
            fn $rn() {
                let r = $p(&super::SCENE);
                r.assert_rule_holds(&super::SCENE, super::$rn);
            }
        )+
    };

    (@red_tests $p:path; { $([$rn:ident : $($rb:tt)+])+ }) => {
        $(
            #[test]
            #[ignore = "TDD の赤。cargo test -- --ignored で、包む前に何が破れるかが見える"]
            fn $rn() {
                let r = $p(&super::SCENE);
                r.assert_rule_holds(&super::SCENE, super::$rn);
            }
        )+
    };

    // --- 宣言順に id を振る (剛体) ---
    (@body_ids $n:expr; { $($items:tt)* }) => { $crate::semantics!(@body_ids_m $n; $($items)*); };
    (@body_ids_m $n:expr;) => {};
    (@body_ids_m $n:expr; [$bn:ident : $($rest:tt)*] $($tail:tt)*) => {
        pub const $bn: usize = $n;
        $crate::semantics!(@body_ids_m $n + 1; $($tail)*);
    };

    // --- 宣言順に id を振る (ゾーン) ---
    (@zone_ids $n:expr; { $($items:tt)* }) => { $crate::semantics!(@zone_ids_m $n; $($items)*); };
    (@zone_ids_m $n:expr;) => {};
    (@zone_ids_m $n:expr; [$zn:ident : $($rest:tt)*] $($tail:tt)*) => {
        pub const $zn: usize = $n;
        $crate::semantics!(@zone_ids_m $n + 1; $($tail)*);
    };

    // --- 宣言順に id を振る (規則) ---
    (@rule_ids $n:expr; { $($items:tt)* }) => { $crate::semantics!(@rule_ids_m $n; $($items)*); };
    (@rule_ids_m $n:expr;) => {};
    (@rule_ids_m $n:expr; [$rn:ident : $($rest:tt)*] $($tail:tt)*) => {
        pub const $rn: u16 = $n;
        $crate::semantics!(@rule_ids_m $n + 1; $($tail)*);
    };

    // --- 表そのもの ---
    (@body_arr {
        $([$bn:ident : axis $ax:literal,
            base ($bx:literal, $by:literal, $bz:literal),
            dir ($dx:literal, $dy:literal, $dz:literal),
            radius $r:literal])+
    }) => {
        &[$(BodyDef {
            name: stringify!($bn),
            axis: $ax,
            base: P3::new($bx, $by, $bz),
            dir: P3::new($dx, $dy, $dz),
            radius_um: $r,
        }),+]
    };

    (@zone_arr {
        $([$zn:ident : id $id:literal,
            center ($cx:literal, $cy:literal, $cz:literal),
            half ($hx:literal, $hy:literal, $hz:literal)])+
    }) => {
        &[$(ZoneDef {
            name: stringify!($zn),
            id: $id,
            aabb: Aabb::centered(P3::new($cx, $cy, $cz), P3::new($hx, $hy, $hz)),
        }),+]
    };

    (@rule_arr { $([$($rule:tt)+])+ }) => {
        &[$($crate::rule_def!($($rule)+)),+]
    };
}
