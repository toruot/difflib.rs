# difflib.rs

Pythonのdifflibを、Rustに移植。

[![Build Status](https://travis-ci.org/toruot/difflib.rs.svg?branch=master)](https://travis-ci.org/toruot/difflib.rs)

参考： https://docs.python.org/3/library/difflib.html (日本語訳: http://docs.python.jp/3.4/library/difflib.html )

注：  
不完全な移植、かつ、改変が入っているため、Pythonのdifflibと同じようには使えない。  
テストパターン少ない。ドキュメント無し。ベンチマーク無し。

# Installation

`Cargo.toml`に以下を追加。
```toml
[dependencies.difflib]
git = "https://github.com/toruot/difflib.rs.git"
```

# Sample

diffコマンドのようなもの。

```toml
[dependencies.difflib]
git = "https://github.com/toruot/difflib.rs"

[dependencies]
docopt = "0.6"
rustc-serialize = "0.3"
```

```rust
extern crate difflib;

extern crate docopt;
extern crate rustc_serialize;

use docopt::Docopt;

const USAGE: &'static str = "
Naval Fate.

Usage:
  diff-rs [options] <file1> <file2>
  diff-rs (-h | --help)

Options:
  -h --help    Show this screen.
  -i           Inner Diff.
";

#[derive(RustcDecodable)]
struct Args {
    flag_i: bool,
    arg_file1: String,
    arg_file2: String,
}

fn main() {
    let a: Args = Docopt::new(USAGE)
                         .and_then(|d| d.decode())
                         .unwrap_or_else(|e| e.exit());

    if a.flag_i {
        difflib::inner_diff(a.arg_file1, a.arg_file2);
    } else {
        difflib::unified_diff(a.arg_file1, a.arg_file2);
    }
}
```
