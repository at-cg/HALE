use std::fs;
use std::path::Path;

fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let dest = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &dest);
        } else {
            fs::copy(&path, &dest).unwrap();
        }
    }
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    // OUT_DIR = target/release/build/hale-xxxx/out
    let target_dir = Path::new(&out_dir)
        .ancestors()
        .nth(4)
        .unwrap();

    let release_dir = target_dir.join("release");

    // ---- install wrapper ----
    let wrapper_src = Path::new("scripts/hale_wrapper.sh");
    let wrapper_dst = release_dir.join("hale");
    fs::copy(wrapper_src, &wrapper_dst).unwrap();

    // ---- install scripts/ ----
    let scripts_src = Path::new("scripts");
    let scripts_dst = release_dir.join("scripts");
    copy_dir(scripts_src, &scripts_dst);

    // ---- chmod +x ----
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        // wrapper
        let mut perms = fs::metadata(&wrapper_dst).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&wrapper_dst, perms).unwrap();

        // scripts
        for entry in fs::read_dir(&scripts_dst).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                let mut perms = fs::metadata(&path).unwrap().permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&path, perms).unwrap();
            }
        }
    }
}
