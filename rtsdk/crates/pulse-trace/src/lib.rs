//! パルス波形の共通スキーマ。
//!
//! 制御側 (`vtime` / `zoneguard` / レガシー橋渡し) が書き、
//! シミュレータが検証し、UI が読む — 3 者が同じ 1 種類のレコードだけを見る。
//!
//! * `no_std` かつ `alloc` を持ち込まない。レコーダは固定長配列で、
//!   容量を超えたぶんは捨てて `dropped` に数える (伸ばさない = WCET が動かない)。
//! * 時刻は仮想時間の µs。実時間のシステムコールはこの層に存在しない。
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

/// パルス 1 本 = レーン (縦軸) のライフサイクル上の 1 イベント。
///
/// `a` / `b` の意味は `kind` で決まる。UI 側はこの表だけ見れば描ける。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Kind {
    /// タスク起動 (立ち上がりエッジ)。`a` = 起動要因コード。
    TaskRise = 1,
    /// タスク消滅 (立ち下がりエッジ)。`a` = 終了コード。
    TaskFall = 2,
    /// 排他ゾーンへの進入。`a` = zone id。
    ZoneEnter = 3,
    /// 排他ゾーンからの退出。`a` = zone id。
    ZoneExit = 4,
    /// 軸位置の更新。`a` = 位置 [µm]。
    AxisPos = 5,
    /// 空間セマンティクス違反。`a` = 規則 id、`b` = 実測値 [µm]。
    Violation = 6,
    /// デッドライン超過。`a` = 予算 [µs]、`b` = 実績 [µs]。
    DeadlineMiss = 7,
    /// 排他トークンの取得。`a` = zone id。
    PermitTake = 8,
    /// 排他トークンの返却。`a` = zone id。
    PermitDrop = 9,
    /// 設計上のマーカ (期待タイムライン側の目盛)。`a` = 任意。
    Mark = 10,
}

impl Kind {
    pub const fn from_u8(v: u8) -> Option<Kind> {
        Some(match v {
            1 => Kind::TaskRise,
            2 => Kind::TaskFall,
            3 => Kind::ZoneEnter,
            4 => Kind::ZoneExit,
            5 => Kind::AxisPos,
            6 => Kind::Violation,
            7 => Kind::DeadlineMiss,
            8 => Kind::PermitTake,
            9 => Kind::PermitDrop,
            10 => Kind::Mark,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Kind::TaskRise => "TaskRise",
            Kind::TaskFall => "TaskFall",
            Kind::ZoneEnter => "ZoneEnter",
            Kind::ZoneExit => "ZoneExit",
            Kind::AxisPos => "AxisPos",
            Kind::Violation => "Violation",
            Kind::DeadlineMiss => "DeadlineMiss",
            Kind::PermitTake => "PermitTake",
            Kind::PermitDrop => "PermitDrop",
            Kind::Mark => "Mark",
        }
    }

    /// 波形を赤くする種類か (UI のハイライト条件)。
    pub const fn is_fault(self) -> bool {
        matches!(self, Kind::Violation | Kind::DeadlineMiss)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ev {
    /// 仮想時間 [µs]。
    pub t_us: u64,
    /// 縦軸のレーン (タスク / 軸 / SDK ライフサイクル) 番号。
    pub lane: u16,
    pub kind: Kind,
    pub a: i32,
    pub b: i32,
}

impl Ev {
    pub const fn new(t_us: u64, lane: u16, kind: Kind, a: i32, b: i32) -> Self {
        Ev {
            t_us,
            lane,
            kind,
            a,
            b,
        }
    }
}

impl Default for Ev {
    fn default() -> Self {
        Ev {
            t_us: 0,
            lane: 0,
            kind: Kind::Mark,
            a: 0,
            b: 0,
        }
    }
}

/// イベントの出口。`vtime::Prog` はこれ越しにしか外へ書けない。
pub trait Sink {
    fn emit(&mut self, ev: Ev);
}

/// 捨てる出口。WCET の見積りに対して「記録が有る / 無い」で差が出ないことを確かめるのに使う。
impl Sink for () {
    fn emit(&mut self, _ev: Ev) {}
}

impl<T: Sink + ?Sized> Sink for &mut T {
    fn emit(&mut self, ev: Ev) {
        (**self).emit(ev)
    }
}

/// 固定長のレコーダ。確保も再確保もしない。
#[derive(Clone, Copy)]
pub struct Recorder<const N: usize> {
    buf: [Ev; N],
    len: usize,
    dropped: u32,
}

impl<const N: usize> Default for Recorder<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Recorder<N> {
    pub const fn new() -> Self {
        Recorder {
            buf: [Ev {
                t_us: 0,
                lane: 0,
                kind: Kind::Mark,
                a: 0,
                b: 0,
            }; N],
            len: 0,
            dropped: 0,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.dropped = 0;
    }

    pub fn events(&self) -> &[Ev] {
        &self.buf[..self.len]
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 容量を超えて捨てた件数。0 でないトレースは UI 上でも「欠落あり」と出す。
    pub const fn dropped(&self) -> u32 {
        self.dropped
    }

    pub fn first_fault(&self) -> Option<Ev> {
        self.events().iter().copied().find(|e| e.kind.is_fault())
    }

    pub fn count(&self, kind: Kind) -> usize {
        self.events().iter().filter(|e| e.kind == kind).count()
    }

    /// UI と外部ツールが読む JSON 行を書き出す。`std` 不要 (`core::fmt::Write` のみ)。
    pub fn write_json(&self, w: &mut impl core::fmt::Write) -> core::fmt::Result {
        w.write_str("{\"dropped\":")?;
        write!(w, "{}", self.dropped)?;
        w.write_str(",\"events\":[")?;
        for (i, e) in self.events().iter().enumerate() {
            if i > 0 {
                w.write_char(',')?;
            }
            write!(
                w,
                "{{\"t_us\":{},\"lane\":{},\"kind\":\"{}\",\"a\":{},\"b\":{}}}",
                e.t_us,
                e.lane,
                e.kind.as_str(),
                e.a,
                e.b
            )?;
        }
        w.write_str("]}")
    }

    /// `proto/pulse.proto` の `Trace` としてエンコードする。
    ///
    /// 依存を足さずに済ませるため varint / zigzag を直に書いている。
    /// 出力先は呼び出し側が持つ固定バッファで、足りなければ `None` を返すだけ —
    /// ここでも確保はしない。
    pub fn encode_proto(&self, out: &mut [u8]) -> Option<usize> {
        let mut n = 0usize;
        for e in self.events() {
            let mut item = [0u8; 40];
            let mut m = 0usize;
            m += put_tag_varint(&mut item[m..], 1, e.t_us)?;
            m += put_tag_varint(&mut item[m..], 2, e.lane as u64)?;
            m += put_tag_varint(&mut item[m..], 3, e.kind as u64)?;
            m += put_tag_varint(&mut item[m..], 4, zigzag(e.a))?;
            m += put_tag_varint(&mut item[m..], 5, zigzag(e.b))?;
            // Trace.events は field 1, wire type 2 (length-delimited)
            n += put_varint(out.get_mut(n..)?, (1 << 3) | 2)?;
            n += put_varint(out.get_mut(n..)?, m as u64)?;
            let dst = out.get_mut(n..n + m)?;
            dst.copy_from_slice(&item[..m]);
            n += m;
        }
        Some(n)
    }
}

impl<const N: usize> Sink for Recorder<N> {
    fn emit(&mut self, ev: Ev) {
        if self.len < N {
            self.buf[self.len] = ev;
            self.len += 1;
        } else {
            self.dropped = self.dropped.saturating_add(1);
        }
    }
}

const fn zigzag(v: i32) -> u64 {
    ((v << 1) ^ (v >> 31)) as u32 as u64
}

fn put_varint(out: &mut [u8], mut v: u64) -> Option<usize> {
    let mut n = 0;
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        let more = v != 0;
        *out.get_mut(n)? = if more { byte | 0x80 } else { byte };
        n += 1;
        if !more {
            return Some(n);
        }
    }
}

fn put_tag_varint(out: &mut [u8], field: u32, v: u64) -> Option<usize> {
    let n = put_varint(out, (field as u64) << 3)?;
    Some(n + put_varint(out.get_mut(n..)?, v)?)
}

#[cfg(feature = "std")]
mod json_in {
    use super::{Ev, Kind};

    /// UI が読み戻すための最小パーサ。`write_json` が出した形だけを受ける。
    ///
    /// 外から拾ってきた任意の JSON を通す口ではない (壊れていれば `None`)。
    pub fn parse_events(s: &str) -> Option<Vec<Ev>> {
        let mut out = Vec::new();
        let body = s.split("\"events\":[").nth(1)?;
        for chunk in body.split('{').skip(1) {
            let obj = chunk.split('}').next()?;
            let mut t_us = 0u64;
            let mut lane = 0u16;
            let mut kind = None;
            let mut a = 0i32;
            let mut b = 0i32;
            for kv in obj.split(',') {
                let (k, v) = kv.split_once(':')?;
                let k = k.trim().trim_matches('"');
                let v = v.trim();
                match k {
                    "t_us" => t_us = v.parse().ok()?,
                    "lane" => lane = v.parse().ok()?,
                    "kind" => {
                        let name = v.trim_matches('"');
                        kind = (1u8..=10)
                            .filter_map(Kind::from_u8)
                            .find(|k| k.as_str() == name);
                    }
                    "a" => a = v.parse().ok()?,
                    "b" => b = v.parse().ok()?,
                    _ => {}
                }
            }
            out.push(Ev {
                t_us,
                lane,
                kind: kind?,
                a,
                b,
            });
        }
        Some(out)
    }
}

#[cfg(feature = "std")]
pub use json_in::parse_events;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_drops_instead_of_growing() {
        let mut r = Recorder::<4>::new();
        for i in 0..7 {
            r.emit(Ev::new(i, 0, Kind::AxisPos, i as i32, 0));
        }
        assert_eq!(r.len(), 4);
        assert_eq!(r.dropped(), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn json_roundtrip() {
        let mut r = Recorder::<8>::new();
        r.emit(Ev::new(10, 1, Kind::TaskRise, 0, 0));
        r.emit(Ev::new(20, 2, Kind::Violation, 7, -1234));
        let mut s = String::new();
        r.write_json(&mut s).unwrap();
        let back = parse_events(&s).unwrap();
        assert_eq!(back, r.events());
    }

    #[test]
    fn proto_encodes_within_a_fixed_buffer() {
        let mut r = Recorder::<2>::new();
        r.emit(Ev::new(1_000_000, 3, Kind::ZoneEnter, 1, 0));
        let mut buf = [0u8; 64];
        let n = r.encode_proto(&mut buf).unwrap();
        assert!(n > 0 && n < 64);
        // 1 件目のヘッダは field 1 / wire type 2。
        assert_eq!(buf[0], 0x0a);
        assert_eq!(buf[1] as usize, n - 2);
        // 溢れる器には書かない。
        let mut tiny = [0u8; 3];
        assert!(r.encode_proto(&mut tiny).is_none());
    }
}
