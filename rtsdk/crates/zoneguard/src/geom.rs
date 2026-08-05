//! 固定小数の 3D 幾何。単位は µm、内部計算は i64。
//!
//! 浮動小数を使わないのは精度の話ではなく、**同じ入力が常に同じビット列を返す**
//! ことを保証するため。丸めが処理系や最適化レベルで動くと、
//! 「テストでは通ったが実機では干渉した」が起きる。

/// マイクロメートル。
pub type Um = i32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct P3 {
    pub x: Um,
    pub y: Um,
    pub z: Um,
}

impl P3 {
    pub const fn new(x: Um, y: Um, z: Um) -> Self {
        P3 { x, y, z }
    }

    pub const ORIGIN: P3 = P3::new(0, 0, 0);

    pub const fn add(self, o: P3) -> P3 {
        P3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    /// 方向ベクトル (成分が -1/0/1) を距離 `k` [µm] だけ伸ばす。
    pub const fn along(dir: P3, k: Um) -> P3 {
        P3::new(dir.x * k, dir.y * k, dir.z * k)
    }
}

/// 2 点間の距離の 2 乗 [µm²]。平方根を取らないので厳密。
pub const fn dist2_um(a: P3, b: P3) -> i64 {
    let dx = (a.x as i64) - (b.x as i64);
    let dy = (a.y as i64) - (b.y as i64);
    let dz = (a.z as i64) - (b.z as i64);
    dx * dx + dy * dy + dz * dz
}

/// 2 点間の距離 [µm]。整数平方根なので切り捨て (安全側に短く出る)。
pub fn dist_um(a: P3, b: P3) -> i64 {
    (dist2_um(a, b) as u64).isqrt() as i64
}

/// 軸並行境界箱。ゾーンの定義に使う。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Aabb {
    pub min: P3,
    pub max: P3,
}

impl Aabb {
    pub const fn new(min: P3, max: P3) -> Self {
        Aabb { min, max }
    }

    /// 中心と半径 (各軸の半幅) から作る。
    pub const fn centered(c: P3, h: P3) -> Self {
        Aabb {
            min: P3::new(c.x - h.x, c.y - h.y, c.z - h.z),
            max: P3::new(c.x + h.x, c.y + h.y, c.z + h.z),
        }
    }

    pub const fn contains(&self, p: P3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// 原点から見た最遠端 (x 方向)。可達長の静的検査に使う。
    pub const fn far_x(&self) -> Um {
        if self.max.x > self.min.x {
            self.max.x
        } else {
            self.min.x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_exact_and_deterministic() {
        let a = P3::new(0, 0, 0);
        let b = P3::new(30_000, 40_000, 0);
        assert_eq!(dist2_um(a, b), 2_500_000_000);
        assert_eq!(dist_um(a, b), 50_000);
    }

    #[test]
    fn aabb_contains_its_boundary() {
        let z = Aabb::centered(P3::new(200_000, 0, 0), P3::new(50_000, 50_000, 50_000));
        assert!(z.contains(P3::new(150_000, 0, 0)));
        assert!(z.contains(P3::new(250_000, 50_000, -50_000)));
        assert!(!z.contains(P3::new(149_999, 0, 0)));
        assert_eq!(z.far_x(), 250_000);
    }
}
