//! 現場の 1 セルを宣言する。
//!
//! ここに書いてあることが全部。図面 (剛体・ゾーン)、守るべき約束 (規則)、
//! そして「どの走らせ方が緑であるべきで、どれが赤の出発点か」。
//! 規則 1 本ごとの `#[test]` は [`semantics!`](crate::semantics) がここから生やす。

use crate::semantics;

semantics! {
    scene cell01 "2 本のアームが 1 つのワークエリアを左右から取り合う" {
        bodies {
            [ARM_A: axis 0, base (0, 0, 0), dir (1, 0, 0), radius 30_000]
            [ARM_B: axis 1, base (400_000, 0, 0), dir (-1, 0, 0), radius 30_000]
        }
        zones {
            [ZONE_X: id 1, center (200_000, 0, 0), half (50_000, 60_000, 60_000)]
        }
        rules {
            [SEPARATION: separation(ARM_A, ARM_B) >= 50_000]
            [EXCLUSIVE_X: exclusive(ZONE_X)]
            [TACT: deadline <= 400_000]
        }
        verify {
            [green typed_sdk = crate::runners::run_typed_sdk]
            [red legacy_raw = crate::runners::run_legacy_raw]
        }
    }
}

semantics! {
    scene cell01_tight "同じセルを、タクト 80 ms で回そうとしたとき" {
        bodies {
            [ARM_A: axis 0, base (0, 0, 0), dir (1, 0, 0), radius 30_000]
            [ARM_B: axis 1, base (400_000, 0, 0), dir (-1, 0, 0), radius 30_000]
        }
        zones {
            [ZONE_X: id 1, center (200_000, 0, 0), half (50_000, 60_000, 60_000)]
        }
        rules {
            [SEPARATION: separation(ARM_A, ARM_B) >= 50_000]
            [EXCLUSIVE_X: exclusive(ZONE_X)]
            [TACT: deadline <= 80_000]
        }
    }
}
