#![feature(convert)]  // as_str()
#![feature(str_char)]  // is_char_boundary()

#![feature(core)]
extern crate core;

use core::cmp::{Ord, Ordering};
use std::collections::HashMap;
use std::hash::Hash;

macro_rules! push_str_ln {
    ( $s:expr, $( $x:expr ),* ) => {
        $(
            $s.push_str($x);
        )*
        $s.push('\n');
    };
}

fn calculate_ratio(matches: usize, length: usize) -> f64 {
    if length > 0 {
        return (2.0 * matches as f64) / length as f64
    }
    return 1.0
}

fn min(a: usize, b: usize) -> usize {
    return if a < b {a} else {b}
}
fn max(a: usize, b: usize, n: usize) -> usize {
    return if n > b { a } else if a > b - n {a} else {b - n}
}

#[derive(PartialEq, Clone, Debug,)]
pub enum OpcodeTag {
    Replace,
    Delete,
    Insert,
    Equal,
}

#[derive(Clone, Debug)]
pub struct Opcode {
    tag: OpcodeTag,
    a1: usize,
    a2: usize,
    b1: usize,
    b2: usize,
}

macro_rules! opcode {
    ($tag:expr, $a1:expr, $a2:expr, $b1:expr, $b2:expr) => {
        Opcode { tag: $tag, a1: $a1, a2: $a2, b1: $b1, b2: $b2 }
    };
}

#[derive(Eq, PartialEq, PartialOrd, Clone, Debug)]
struct Match {
    i: usize,
    j: usize,
    size: usize,
}

impl Ord for Match {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.i != other.i { return self.i.cmp(&other.i) }
        if self.j != other.j { return self.j.cmp(&other.j) }
        return self.size.cmp(&other.size)
    }
}

#[derive(Debug)]
struct Range {
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
}

struct SequenceMatcher<'l, T: 'l + Eq + Hash + ?Sized> {
    a: &'l Vec<&'l T>,
    b: &'l Vec<&'l T>,

    b2j: HashMap<&'l T, Vec<usize>>,

    fullbcount: HashMap<&'l T, usize>,
    fullbcount_ready: bool,
    matching_blocks: Vec<Match>,
    matching_blocks_ready: bool,
    opcodes: Vec<Opcode>,
    opcodes_ready: bool,

    autojunk: bool,
}
// Members:
//   isjunk
//   bjunk
//   bpopular  unused

impl<'l, T: Eq + Hash + ?Sized> SequenceMatcher<'l, T> {
    pub fn new(aa: &'l Vec<&'l T>, bb: &'l Vec<&'l T>) -> SequenceMatcher<'l, T> {  // (isjunk=None, autojunk=True)
        let mut myself = SequenceMatcher {
            a: aa,
            b: bb,
            b2j: HashMap::new(),
            fullbcount: HashMap::new(),
            fullbcount_ready: false,
            matching_blocks: Vec::new(),
            matching_blocks_ready: false,
            opcodes: Vec::new(),
            opcodes_ready: false,
            autojunk: true,
        };
        // self.isjunk = isjunk

        myself.chain_b();

        return myself
    }
    /*
    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)
    */
    pub fn set_seq1(&mut self, aa: &'l Vec<&'l T>) {
        // if a is self.a:
        //     return
        self.a = aa;

        self.matching_blocks.clear();
        self.matching_blocks_ready = false;
        self.opcodes.clear();
        self.opcodes_ready = false;
    }
    pub fn set_seq2(&mut self, bb: &'l Vec<&'l T>) {
        // if b is self.b:
        //     return
        self.b = bb;

        self.fullbcount.clear();
        self.fullbcount_ready = false;
        self.matching_blocks.clear();
        self.matching_blocks_ready = false;
        self.opcodes.clear();
        self.opcodes_ready = false;

        self.b2j.clear();
        self.chain_b();
    }
    fn chain_b(&mut self) {
        for i in 0 .. self.b.len() {
            if ! self.b2j.contains_key(self.b[i]) {
                self.b2j.insert(self.b[i], vec![]);
            }
            self.b2j.get_mut(self.b[i]).unwrap().push(i);
        }

        /*
        self.bjunk = junk = set()
        isjunk = self.isjunk
        if isjunk:
            for elt in self.b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk: # separate loop avoids separate list of keys
                del self.b2j[elt]
        */

        let mut popular = Vec::new();
        if self.autojunk && self.b.len() >= 200 {
            let ntest = self.b.len() / 100 + 1;
            for (elt, idxs) in self.b2j.iter() {
                if idxs.len() > ntest {
                    popular.push(elt.clone());
                }
            }
            for elt in popular {
                self.b2j.remove(elt);
            }
        }
    }

    fn find_longest_match(&self, r: &Range) -> Match {
        // isbjunk = self.bjunk.__contains__
        let mut best = Match {i: r.alo, j: r.blo, size: 0};
        let mut j2len: HashMap<i64, usize> = HashMap::new();
        for i in r.alo .. r.ahi {
            let mut new_j2len = HashMap::new();
            if self.b2j.contains_key(self.a[i]) {
                for j in self.b2j.get(self.a[i]).unwrap() {
                    if j <  &r.blo { continue; }
                    if &r.bhi <= j { break; }

                    let jm1 = (j + 0) as i64 - 1;

                    if ! j2len.contains_key(&jm1) {
                        j2len.insert(jm1, 0);
                    }
                    *(j2len.get_mut(&jm1).unwrap()) += 1;
                    let s = j2len.get(&jm1).unwrap();

                    new_j2len.insert((j + 0) as i64, s + 0);

                    if s > &best.size {
                        best.i    = (i + 1) - s;
                        best.j    = (j + 1) - s;
                        best.size = s + 0;
                    }
                }
            }
            j2len = new_j2len;
        }

        while best.i > r.alo && best.j > r.blo &&
              // not isbjunk(self.b[bestj-1]) &&
              self.a[best.i - 1] == self.b[best.j - 1] {
            best.i -= 1;
            best.j -= 1;
            best.size += 1;
        }
        while best.i + best.size < r.ahi &&
              best.j + best.size < r.bhi &&
              // not isbjunk(self.b[bestj+bestsize]) &&
              self.a[best.i + best.size] == self.b[best.j + best.size] {
            best.size += 1;
        }

        while best.i > r.alo && best.j > r.blo &&
              // isbjunk(self.b[bestj-1]) &&
              self.a[best.i - 1] == self.b[best.j - 1] {
            best.i -= 1;
            best.j -= 1;
            best.size += 1;
        }
        while best.i + best.size < r.ahi &&
              best.j + best.size < r.bhi &&
              // isbjunk(self.b[bestj+bestsize]) &&
              self.a[best.i + best.size] == self.b[best.j + best.size] {
            best.size += 1;
        }

        return Match{i: best.i, j: best.j, size: best.size}
    }

    fn get_matching_blocks(&mut self) -> Vec<Match> {
        if self.matching_blocks_ready {
            return self.matching_blocks.clone()
        }

        let mut queue = vec![Range{alo: 0, ahi: self.a.len(), blo: 0, bhi: self.b.len()}];
        let mut matching_blocks = vec![];
        while ! queue.is_empty() {
            let r = queue.pop().unwrap();
            let x = self.find_longest_match(&r);
            if x.size > 0 {
                matching_blocks.push(x.clone());
                if r.alo < x.i && r.blo < x.j {
                    queue.push(Range{alo: r.alo, ahi: x.i, blo: r.blo, bhi: x.j});
                }
                if x.i + x.size < r.ahi && x.j + x.size < r.bhi {
                    queue.push(Range{alo: x.i + x.size, ahi: r.ahi, blo: x.j + x.size, bhi: r.bhi});
                }
            }
        }
        matching_blocks.sort();

        let mut x = Match{i: 0, j: 0, size: 0};
        for y in matching_blocks {
            if x.i + x.size == y.i && x.j + x.size == y.j {
                x.size += y.size
            } else {
                if x.size > 0 {
                    self.matching_blocks.push(Match{i: x.i, j: x.j, size: x.size});
                }
                x.i    = y.i;
                x.j    = y.j;
                x.size = y.size;
            }
        }
        if x.size > 0 {
            self.matching_blocks.push(Match{i: x.i, j: x.j, size: x.size});
        }

        self.matching_blocks.push(Match{i: self.a.len(), j: self.b.len(), size: 0});

        self.matching_blocks_ready = true;
        return self.matching_blocks.clone()
    }

    pub fn get_opcodes(&mut self) -> Vec<Opcode> {
        if self.opcodes_ready {
            return self.opcodes.clone()
        }

        let mut i = 0;
        let mut j = 0;
        for x in self.get_matching_blocks() {
            let t =
                if i < x.i && j < x.j { OpcodeTag::Replace
                } else if i < x.i {     OpcodeTag::Delete
                } else if j < x.j {     OpcodeTag::Insert
                } else {                OpcodeTag::Equal
                };
            if t != OpcodeTag::Equal {
                self.opcodes.push(opcode!(t, i, x.i, j, x.j));
            }
            i = x.i + x.size;
            j = x.j + x.size;
            if x.size > 0 {
                self.opcodes.push(opcode!(OpcodeTag::Equal, x.i, i, x.j, j));
            }
        }

        self.opcodes_ready = true;
        return self.opcodes.clone()
    }

    pub fn get_grouped_opcodes(&mut self) -> Vec<Vec<Opcode>> {  // (n=3)
        let n = 3;

        let mut c = self.get_opcodes();
        if c.len() == 0 {
            c.push(opcode!(OpcodeTag::Equal, 0, 1, 0, 1));
        }
        if c[0].tag == OpcodeTag::Equal {
            c[0].a1 = max(c[0].a1, c[0].a2, n);
            c[0].b1 = max(c[0].b1, c[0].b2, n);
        }
        let m1 = c.len() - 1;
        if c[m1].tag == OpcodeTag::Equal {
            c[m1].a2 = min(c[m1].a2, c[m1].a1 + n);
            c[m1].b2 = min(c[m1].b2, c[m1].b1 + n);
        }

        let nn = n + n;
        let mut group = Vec::new();
        let mut codes = Vec::new();
        for cc in c {
            let mut next_c = cc.clone();
            if cc.tag == OpcodeTag::Equal && cc.a2 - cc.a1 > nn {
                codes.push(opcode!(cc.tag,
                                   cc.a1, min(cc.a2, cc.a1 + n),
                                   cc.b1, min(cc.b2, cc.b1 + n)));

                group.push(codes);
                codes = Vec::new();

                next_c.a1 = max(cc.a1, cc.a2, n);
                next_c.b1 = max(cc.b1, cc.b2, n);
            }
            codes.push(next_c);
        }
        if codes.len() > 0 && ! (codes.len() == 1 && codes[0].tag == OpcodeTag::Equal) {
            group.push(codes);
        }

        return group
    }

    pub fn ratio(&mut self) -> f64 {
        let mut matches = 0;
        for x in self.get_matching_blocks() {
            matches += x.size;
        }
        return calculate_ratio(matches, self.a.len() + self.b.len())
    }

    pub fn quick_ratio(&mut self) -> f64 {
        if ! self.fullbcount_ready {
            for elt in self.b {
                if ! self.fullbcount.contains_key(elt) {
                    self.fullbcount.insert(elt, 0);
                }
                *(self.fullbcount.get_mut(elt).unwrap()) += 1;
            }

            self.fullbcount_ready = true;
        }

        let mut matches = 0;
        let mut avail: HashMap<&T, usize> = HashMap::new();
        for elt in self.a {
            if ! self.fullbcount.contains_key(elt) {
                continue;
            }
            if ! avail.contains_key(elt) {
                avail.insert(elt, self.fullbcount.get(elt).unwrap() + 0);
            }
            if avail.get(elt).unwrap() > &0usize {
                matches += 1;
                *(avail.get_mut(elt).unwrap()) -= 1;
            }
        }

        return calculate_ratio(matches, self.a.len() + self.b.len())
    }

    pub fn real_quick_ratio(&self) -> f64 {
        let la = self.a.len();
        let lb = self.b.len();
        return calculate_ratio(min(la, lb), la + lb)
    }
}

fn format_range_unified(start: usize, stop: usize) -> String {
    let mut beginning = start + 1;
    let length = stop - start;

    if length == 1 {
        return format!("{}", beginning)
    }

    if length < 1 {
        beginning -= 1;
    }
    return format!("{},{}", beginning, length)
}

fn set_unified_diff(a: &Vec<&str>, b: &Vec<&str>, s: &mut String) {
    let mut sm = SequenceMatcher::new(a, b);

    for g in sm.get_grouped_opcodes() {
        let m1 = g.len() - 1;
        push_str_ln!(s,
            format!("@@ -{} +{} @@", format_range_unified(g[0].a1, g[m1].a2),
                                     format_range_unified(g[0].b1, g[m1].b2)
                   ).as_str());

        for c in g {
            if c.tag == OpcodeTag::Equal {
                for i in c.a1 .. c.a2 {
                    push_str_ln!(s, " ", a[i]);
                }
            }
            if c.tag == OpcodeTag::Delete || c.tag == OpcodeTag::Replace {
                for i in c.a1 .. c.a2 {
                    push_str_ln!(s, "-", a[i]);
                }
            }
            if c.tag == OpcodeTag::Insert || c.tag == OpcodeTag::Replace {
                for j in c.b1 .. c.b2 {
                    push_str_ln!(s, "+", b[j]);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------

trait SplitChars {
    fn split_chars<'a>(&'a self) -> Vec<&'a str>;
}
impl SplitChars for str {
    fn split_chars<'a>(&'a self) -> Vec<&'a str> {
        let mut vs: Vec<&'a str> = Vec::new();
        let mut start = 0;
        for i in 1 .. self.len() + 1 {
            if ! self.is_char_boundary(i) {
                continue;
            }
            let slice = unsafe { self.slice_unchecked(start, i) };
            vs.push(slice);
            start = i;
        }
        vs
    }
}

#[derive(PartialEq, Debug,)]
pub enum LineType {
    Minus,
    Plus,
    Replace,
    Equal,
    Diff,
}

#[derive(Debug,)]
pub struct LineOpcode {
    line_type: LineType,
    aic: Vec<(OpcodeTag, usize, usize)>,
    bjc: Vec<(OpcodeTag, usize, usize)>,
    i: usize,
    j: usize,
}
macro_rules! line_opcode {
    ($line_type:expr, $aic:expr, $bjc:expr, $i:expr, $j:expr) => {
        LineOpcode { line_type: $line_type, aic: $aic, bjc: $bjc, i: $i, j: $j }
    };
}

pub struct InnerDiffer {
    best_ratio: f64,
    cutoff: f64,
    char_chunk_min: usize,
}

impl InnerDiffer {
    pub fn new() -> InnerDiffer {
        return InnerDiffer{
            best_ratio: 0.48,
            cutoff:     0.49,
            char_chunk_min: 8,
        }
    }

    pub fn compare(&self, pre_a: &Vec<&str>, pre_b: &Vec<&str>, r: &mut Vec<LineOpcode>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for c in pre_a {
            a.push(c.split_chars());
        }
        for c in pre_b {
            b.push(c.split_chars());
        }

        self.fancy_replace(&a, 0, a.len(), pre_a,
                           &b, 0, b.len(), pre_b, r);
    }

    fn fancy_replace(&self, a: &Vec<Vec<&str>>, alo: usize, ahi: usize, pre_a: &Vec<&str>,
                            b: &Vec<Vec<&str>>, blo: usize, bhi: usize, pre_b: &Vec<&str>, r: &mut Vec<LineOpcode>) {
        let mut best_ratio = self.best_ratio;

        let dummy_v1: Vec<&str> = Vec::new();
        let dummy_v2: Vec<&str> = Vec::new();
        let mut cruncher = SequenceMatcher::new(&dummy_v1, &dummy_v2);

        let mut eqi: i64 = -1;
        let mut eqj: i64 = -1;
        let mut best_i: usize = 0;
        let mut best_j: usize = 0;

        for j in blo .. bhi {
            cruncher.set_seq2(&b[j]);
            for i in alo .. ahi {
                if pre_a[i] == pre_b[j] {
                    if eqi < 0 {
                        eqi = i as i64;
                        eqj = j as i64;
                    }
                    continue;
                }
                cruncher.set_seq1(&a[i]);
                if cruncher.real_quick_ratio() > best_ratio &&
                   cruncher.quick_ratio()      > best_ratio &&
                   cruncher.ratio()            > best_ratio {
                    best_ratio = cruncher.ratio();
                    best_i = i;
                    best_j = j;
                    continue;
                }
                if best_ratio < self.cutoff &&
                   ((a[i].len() >= self.char_chunk_min && pre_b[j].find(pre_a[i]).is_some()) ||
                    (b[j].len() >= self.char_chunk_min && pre_a[i].find(pre_b[j]).is_some())) {
                    best_ratio = self.cutoff;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if best_ratio < self.cutoff {
            if eqi < 0 {
                self.plain_replace(alo, ahi, blo, bhi, r);
                return;
            }
            // best_ratio = 1.0;  never read
            best_i = eqi as usize;
            best_j = eqj as usize;
        } else {
            eqi = -1;
        }

        self.fancy_helper(a, alo, best_i, pre_a, b, blo, best_j, pre_b, r);

        if eqi < 0 {
            self.diff_line(best_i, best_j, &a[best_i], &b[best_j], r);
        } else {
            self.equa_line(best_i, r);
        }

        self.fancy_helper(a, best_i + 1, ahi, pre_a, b, best_j + 1, bhi, pre_b, r);
    }

    fn fancy_helper(&self, a: &Vec<Vec<&str>>, alo: usize, ahi: usize, pre_a: &Vec<&str>,
                           b: &Vec<Vec<&str>>, blo: usize, bhi: usize, pre_b: &Vec<&str>, r: &mut Vec<LineOpcode>) {
        if alo < ahi {
            if blo < bhi {
                self.fancy_replace(a, alo, ahi, pre_a, b, blo, bhi, pre_b, r);
            } else {
                self.minu_lines(alo, ahi, r);
            }
        } else if blo < bhi {
            self.plus_lines(blo, bhi, r);
        }
    }

    fn plain_replace(&self, alo: usize, ahi: usize,
                            blo: usize, bhi: usize, r: &mut Vec<LineOpcode>) {
        assert!(alo < ahi || blo < bhi);
        let alen = ahi - alo;
        let blen = bhi - blo;
        let range_max = if alen > blen {alen} else {blen};
        for i in 0 .. range_max {
            if blen <= i && i < alen {
                self.minu_line(alo + i, r);
            } else if alen <= i && i < blen {
                self.plus_line(blo + i, r);
            } else {
                self.repl_line(alo + i, blo + i, r);
            }
        }
    }

    fn minu_lines(&self, alo: usize, ahi: usize, r: &mut Vec<LineOpcode>) {
        for i in alo .. ahi {
            self.minu_line(i, r);
        }
    }
    fn plus_lines(&self, blo: usize, bhi: usize, r: &mut Vec<LineOpcode>) {
        for j in blo .. bhi {
            self.plus_line(j, r);
        }
    }

    fn minu_line(&self, i: usize, r: &mut Vec<LineOpcode>) {
        r.push(line_opcode!(LineType::Minus, vec![], vec![], i, 0));
    }
    fn plus_line(&self, j: usize, r: &mut Vec<LineOpcode>) {
        r.push(line_opcode!(LineType::Plus, vec![], vec![], 0, j));
    }
    fn repl_line(&self, i: usize, j: usize, r: &mut Vec<LineOpcode>) {
        r.push(line_opcode!(LineType::Replace, vec![], vec![], i, j));
    }
    fn equa_line(&self, i: usize, r: &mut Vec<LineOpcode>) {
        r.push(line_opcode!(LineType::Equal, vec![], vec![], i, 0));
    }
    fn diff_line(&self, i: usize, j: usize, ai: &Vec<&str>, bj: &Vec<&str>, r: &mut Vec<LineOpcode>) {
        let mut aic = Vec::new();
        let mut bjc = Vec::new();
        let mut cruncher = SequenceMatcher::new(ai, bj);
        for c in cruncher.get_opcodes() {
            if c.tag != OpcodeTag::Insert {
                aic.push((c.tag.clone(), c.a1, c.a2));
            }
            if c.tag != OpcodeTag::Delete {
                bjc.push((c.tag.clone(), c.b1, c.b2));
            }
        }
        r.push(line_opcode!(LineType::Diff, aic, bjc, i, j));
    }
}

fn set_inner_diff(a: &Vec<&str>, b: &Vec<&str>, s: &mut String) {
    let mut sm = SequenceMatcher::new(a, b);
    let mut buff_old = Vec::new();
    let mut buff_new = Vec::new();

    let differ = InnerDiffer::new();

    for g in sm.get_grouped_opcodes() {
        let m1 = g.len() - 1;
        push_str_ln!(s,
            format!("@@ -{} +{} @@", format_range_unified(g[0].a1, g[m1].a2),
                                     format_range_unified(g[0].b1, g[m1].b2)
                   ).as_str());

        for c in g {
            if c.tag == OpcodeTag::Equal {
                clear_buffer(&differ, &mut buff_old, &mut buff_new, s);
                for i in c.a1 .. c.a2 {
                    push_str_ln!(s, "  |", a[i]);
                }
            }
            if c.tag == OpcodeTag::Replace || c.tag == OpcodeTag::Delete {
                for i in c.a1 .. c.a2 {
                    buff_old.push(a[i]);
                }
            }
            if c.tag == OpcodeTag::Replace || c.tag == OpcodeTag::Insert {
                for j in c.b1 .. c.b2 {
                    buff_new.push(b[j]);
                }
            }
        }
        clear_buffer(&differ, &mut buff_old, &mut buff_new, s);
    }
}
fn clear_buffer(differ: &InnerDiffer, buff_old: &mut Vec<&str>, buff_new: &mut Vec<&str>, s: &mut String) {
    if buff_old.len() == 0 && buff_new.len() == 0 {
        return
    }

    struct Result<'l>(&'l str, Option<Vec<(OpcodeTag, usize, usize)>>);

    let mut olds = Vec::new();
    let mut news = Vec::new();
    let mut result: Vec<LineOpcode> = Vec::new();
    differ.compare(buff_old, buff_new, &mut result);
    for c in result {
        if c.line_type == LineType::Replace || c.line_type == LineType::Minus {
            olds.push(Result(buff_old[c.i], None));
        }
        if c.line_type == LineType::Replace || c.line_type == LineType::Plus {
            news.push(Result(buff_new[c.j], None));
        }
        if c.line_type == LineType::Diff {
            olds.push(Result(buff_old[c.i], Some(c.aic)));
            news.push(Result(buff_new[c.j], Some(c.bjc)));
        } else if c.line_type == LineType::Equal {
            olds.push(Result(buff_old[c.i], Some(c.aic)));
            news.push(Result(buff_old[c.i], Some(c.bjc)));
        }
    }

    for Result(l, v) in olds {
        push_str_ln!(s, "--|", l);
        if v.is_some() {
            set_marker(v.unwrap(), l.chars().collect(), s);
        }
    }
    for Result(l, v) in news {
        push_str_ln!(s, "++|", l);
        if v.is_some() {
            set_marker(v.unwrap(), l.chars().collect(), s);
        }
    }

    buff_old.clear();
    buff_new.clear();
}
fn set_marker(v: Vec<(OpcodeTag, usize, usize)>, lv: Vec<char>, s: &mut String) {
    s.push_str(" >|");
    for (tag, start, end) in v {
        let c = match tag {
            OpcodeTag::Equal => ' ',
            _ => '^',
        };
        for i in start .. end {
            s.push(c);
            if lv[i].len_utf8() == 3 {
                s.push(c);
            }
        }
    }
    s.push('\n');
}

// -----------------------------------------------------------------------------

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn read_file(path_str: &String) -> String {
    let path = Path::new(path_str);
    let mut file = File::open(&path).unwrap();
    let mut string = String::new();

    let _ = file.read_to_string(&mut string);

    string
}

fn print_some_diff(path_a: String, path_b: String, some_diff: fn(&Vec<&str>, &Vec<&str>, &mut String)) {
    let string_a = read_file(&path_a);
    let string_b = read_file(&path_b);
    let vec_a: Vec<&str> = string_a.lines().collect();
    let vec_b: Vec<&str> = string_b.lines().collect();

    let mut diff_string = String::new();
    some_diff(&vec_a, &vec_b, &mut diff_string);

    println!("--- {}", path_a);
    println!("+++ {}", path_b);
    print!("{}", diff_string);
}

pub fn unified_diff(path_a: String, path_b: String) {
    print_some_diff(path_a, path_b, set_unified_diff);
}

pub fn inner_diff(path_a: String, path_b: String) {
    print_some_diff(path_a, path_b, set_inner_diff);
}
