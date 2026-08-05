fn main() {
    println!("cargo:rerun-if-changed=cxx/arm_ctl_1998.cpp");
    cc::Build::new()
        .cpp(true)
        .file("cxx/arm_ctl_1998.cpp")
        // 資産は資産のまま通す。警告を潰すためにソースへ手を入れない。
        .warnings(false)
        .compile("arm_ctl_1998");
}
