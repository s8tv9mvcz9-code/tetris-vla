/*
 * arm_ctl_1998.cpp  —  設備更新のたびに移植されてきた、現役の資産という設定。
 *
 * 触ってはいけない前提で扱う。作った人はもう居ないが、
 * 「軸を 1 ステップずつ送る」内側のループは実機で 20 年動いていて、
 * 加減速もサーボの癖もここに畳み込まれている。捨てる理由が無い。
 *
 * 捨てたいのは外側 — ゾーン X の排他を g_busy 1 個でやっているところ。
 * 症状は「稀に 2 本のアームが同時にゾーンへ入る」。
 * 原因は下のコメントに書いたとおりで、直し方も分かっているが、
 * このファイルを直しても「もう起きない」ことは証明できない。
 * 証明できる場所へ持っていくのが rtsdk の仕事で、
 * その入口としてこのファイルは **そのまま** 残す。
 *
 * 唯一の改造は、usleep() と直接のサーボ書き込みを関数ポインタに差し替えたこと。
 * これで実時間のシステムコールが消え、外から仮想時間を注入できる。
 * ロジックには 1 行も触っていない。
 */

#include <stdint.h>
#include <stddef.h>

typedef void (*rt_sleep_fn)(uint32_t us);
typedef void (*rt_axis_fn)(int32_t axis, int32_t pos_um);
typedef void (*rt_mark_fn)(int32_t axis, int32_t code);

static rt_sleep_fn g_sleep = NULL;
static rt_axis_fn g_write = NULL;
static rt_mark_fn g_mark = NULL;

extern "C" void legacy_install_hooks(rt_sleep_fn s, rt_axis_fn w, rt_mark_fn m) {
    g_sleep = s;
    g_write = w;
    g_mark = m;
}

#define AXIS_MAX 2
#define STEP_UM 2000       /* 1 送りの移動量 [µm] */
#define STEP_SLEEP_US 200  /* 1 送りの待ち。元は usleep(200) */

static int32_t g_pos[AXIS_MAX] = {0, 0};
static int32_t g_busy = 0; /* ゾーン X 使用中フラグ。これで排他しているつもり */

extern "C" void legacy_reset(void) {
    g_pos[0] = 0;
    g_pos[1] = 0;
    g_busy = 0;
}

extern "C" int32_t legacy_axis_pos(int32_t axis) {
    if (axis < 0 || axis >= AXIS_MAX) return 0;
    return g_pos[axis];
}

extern "C" int32_t legacy_zone_busy(void) { return g_busy; }

/*
 * 内側の送りループ。ここは残す資産。
 */
extern "C" void legacy_arm_move(int32_t axis, int32_t target_um) {
    if (axis < 0 || axis >= AXIS_MAX) return;
    while (g_pos[axis] != target_um) {
        int32_t d = target_um - g_pos[axis];
        int32_t s;
        if (d > 0) {
            s = (d < STEP_UM) ? d : STEP_UM;
        } else {
            s = (-d < STEP_UM) ? d : -STEP_UM;
        }
        g_pos[axis] += s;
        if (g_write) g_write(axis, g_pos[axis]);
        if (g_sleep) g_sleep(STEP_SLEEP_US);
    }
}

/*
 * 外側のピック手順。ここが問題。
 *
 *   (1) g_busy を見る
 *   (2) 空いていなければ 1 ms 待つ … だけで、待った後にもう一度見ない
 *   (3) g_busy = 1 を書く
 *
 * (1) と (3) の間には、送りループの待ちを挟んで他軸が何度でも割り込める。
 * つまりこれは排他ではなく、「たいてい当たる時間差」でしかない。
 * タクトを詰めた日、あるいはサーボを速い型に替えた日に、当たらなくなる。
 */
extern "C" void legacy_pick_sequence(int32_t axis, int32_t zone_um, uint32_t hold_us) {
    if (g_busy) {
        if (g_sleep) g_sleep(1000); /* ← 待つだけ。再確認が無い */
    }
    g_busy = 1;

    if (g_mark) g_mark(axis, 1);
    legacy_arm_move(axis, zone_um);
    if (g_sleep) g_sleep(hold_us);
    legacy_arm_move(axis, 0);
    if (g_mark) g_mark(axis, 0);

    g_busy = 0;
}
