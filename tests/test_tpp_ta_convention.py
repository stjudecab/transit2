"""Tests for the TA convention of tpp (is the target-site TA trimmed together with the primer?).

Self-contained: genomes, alignments and reads are generated on the fly, so no data files are needed.
The end-to-end tests need bwa on the PATH and are skipped without it.
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

basedir = os.path.dirname(__file__)
sys.path.insert(0, basedir + "/../src/")

from pytpp.tpp_tools import (
    Globals,
    clean_args,
    cigar_reference_span,
    describe_ta_convention,
    describe_wig_sites,
    initialize_globals,
    junction_site,
    read_counts,
    resolve_ta_convention,
    resolve_wig_sites,
    site_positions,
    template_counts,
)

ITR = "GAAGACCGGGGACTTATCATCCAACCTGT"  # Himar1 ITR, without the target-site TA
DEFAULT_PRIMER = "ACTTATCAGCCAACCTGTTA"  # tpp's default Himar1 primer, ends in TA
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq):
    return seq.translate(COMP)[::-1]


def make_vars(prefix, convention="auto", transposon="Himar1", protocol="Sassetti", num_replicons=1, wig_sites="ta"):
    v = Globals()
    v.transposon = transposon
    v.protocol = protocol
    v.prefix = prefix
    v.ta_convention = convention
    v.wig_sites = wig_sites
    v.num_replicons = num_replicons
    v.allow_improper = False
    v.proper_read_pairs = 0
    v.improper_read_pairs = 0
    return v


def ta_sites(genome, margin=100, n=8):
    """The 1-based coordinate of the T of n TA dinucleotides away from the genome ends."""
    sites = [i + 1 for i in range(margin, len(genome) - margin) if genome[i:i + 2] == "TA"]
    return sites[:: max(1, len(sites) // n)][:n]


class TestHelpers(unittest.TestCase):

    def test_auto_infers_convention_from_primer(self):
        self.assertEqual(resolve_ta_convention(make_vars(DEFAULT_PRIMER)), "trimmed")
        self.assertEqual(resolve_ta_convention(make_vars(DEFAULT_PRIMER.lower())), "trimmed")
        self.assertEqual(resolve_ta_convention(make_vars(ITR)), "retained")
        self.assertEqual(resolve_ta_convention(make_vars(ITR + "TA")), "trimmed")

    def test_empty_primer_keeps_the_historical_arithmetic(self):
        self.assertEqual(resolve_ta_convention(make_vars("")), "trimmed")
        self.assertEqual(resolve_ta_convention(make_vars("", convention="retained")), "retained")

    def test_explicit_choice_wins(self):
        self.assertEqual(resolve_ta_convention(make_vars(DEFAULT_PRIMER, convention="retained")), "retained")
        self.assertEqual(resolve_ta_convention(make_vars(ITR, convention="trimmed")), "trimmed")

    def test_only_ta_based_himar1_counting_is_affected(self):
        self.assertEqual(resolve_ta_convention(make_vars(ITR, transposon="Tn5", protocol="Tn5")), "trimmed")
        self.assertEqual(resolve_ta_convention(make_vars(ITR, convention="retained", transposon="Tn5", protocol="Tn5")), "trimmed")
        self.assertEqual(resolve_ta_convention(make_vars(ITR, protocol="Mme1")), "trimmed")

    def test_objects_without_the_new_attributes_behave_as_before(self):
        v = Globals()
        self.assertEqual(resolve_ta_convention(v), "trimmed")

    def test_description_is_informative(self):
        self.assertIn("auto: the primer does not end in TA", describe_ta_convention(make_vars(ITR)))
        self.assertIn("auto: the primer ends in TA", describe_ta_convention(make_vars(DEFAULT_PRIMER)))
        self.assertIn("-ta-convention retained", describe_ta_convention(make_vars("")))
        self.assertIn("set with -ta-convention", describe_ta_convention(make_vars(ITR, convention="trimmed")))

    def test_cigar_reference_span(self):
        self.assertEqual(cigar_reference_span("50M", 50), 50)
        self.assertEqual(cigar_reference_span("30M2D20M", 50), 52)
        self.assertEqual(cigar_reference_span("25M3I22M", 50), 47)
        self.assertEqual(cigar_reference_span("5S40M5S", 50), 40)
        self.assertEqual(cigar_reference_span("10M100N10M", 20), 120)
        self.assertEqual(cigar_reference_span("*", 33), 33)

    def test_junction_site(self):
        # trimmed: the aligned read starts right after the TA
        self.assertEqual(junction_site("trimmed", False, 102, 50), 100)
        self.assertEqual(junction_site("trimmed", True, 50, 50), 100)
        # retained: the TA is the first two bases of the aligned read
        self.assertEqual(junction_site("retained", False, 100, 50), 100)
        self.assertEqual(junction_site("retained", True, 52, 50), 100)

    def test_option_parsing(self):
        v = Globals()
        initialize_globals(v, kwargs={"himar1": True})
        self.assertEqual(v.ta_convention, "auto")
        for choice in ("auto", "trimmed", "retained", "Retained"):
            v = Globals()
            args, kwargs = clean_args(["-ta-convention", choice])
            initialize_globals(v, kwargs=kwargs)
            self.assertEqual(v.ta_convention, choice.lower())
        v = Globals()
        args, kwargs = clean_args(["-ta-convention", "sideways"])
        with self.assertRaises(ValueError):
            initialize_globals(v, kwargs=kwargs)

    def test_primer_flows_into_the_resolved_convention(self):
        for primer, expected in ((None, "trimmed"), (DEFAULT_PRIMER, "trimmed"), (ITR, "retained"), ("", "trimmed")):
            v = Globals()
            kwargs = {"himar1": True}
            if primer is not None:
                kwargs["primer"] = primer
            initialize_globals(v, kwargs=kwargs)
            self.assertEqual(resolve_ta_convention(v), expected, primer)


class TestCounting(unittest.TestCase):
    """read_counts / template_counts on hand-made alignments (single replicon)."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(7)
        cls.genome = "".join(rng.choice("ACGT") for _ in range(4000))
        cls.sites = ta_sites(cls.genome)
        cls.tmp = tempfile.mkdtemp(prefix="tpp_ta_conv_")
        cls.fasta = os.path.join(cls.tmp, "ref.fa")
        with open(cls.fasta, "w") as fh:
            fh.write(">chr\n" + cls.genome + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def write_sam(self, records):
        path = os.path.join(self.tmp, "aln.sam")
        with open(path, "w") as fh:
            fh.write("@HD\tVN:1.6\n@SQ\tSN:chr\tLN:%d\n" % len(self.genome))
            for name, flag, pos, mapq, cigar, seq_len in records:
                fh.write("\t".join([name, str(flag), "chr", str(pos), str(mapq), cigar, "*", "0", "0", "A" * seq_len, "I" * seq_len]) + "\n")
        return path

    @staticmethod
    def hit_sites(counts):
        return {row[0]: tuple(row[1:7]) for row in counts[0] if row[5] > 0}

    def test_retained_convention_single_end(self):
        t1, t2, t3, t4, t5 = self.sites[:5]
        sam = self.write_sam([
            ("f1", 0, t1, 60, "50M", 50),
            ("f2", 0, t1, 60, "50M", 50),
            ("r_plain", 16, t2 + 1 - 49, 60, "50M", 50),          # alignment ends on the A of the TA
            ("r_del", 16, t3 + 1 - 51, 60, "30M2D20M", 50),        # span 52, SEQ 50
            ("r_ins", 16, t4 + 1 - 46, 60, "25M3I22M", 50),        # span 47, SEQ 50
            ("off", 0, t5 + 1, 60, "50M", 50),                     # not a TA
            ("unm", 4, 0, 0, "*", 50),
        ])
        counts = read_counts(self.fasta, sam, make_vars(ITR))     # auto -> retained
        self.assertEqual(self.hit_sites(counts), {
            t1: (2, 2, 0, 0, 2, 2),
            t2: (0, 0, 1, 1, 1, 1),
            t3: (0, 0, 1, 1, 1, 1),
            t4: (0, 0, 1, 1, 1, 1),
        })

    def test_trimmed_convention_single_end(self):
        t1, t2, t3 = self.sites[:3]
        sam = self.write_sam([
            ("f1", 0, t1 + 2, 60, "50M", 50),                      # starts right after the TA
            ("r_plain", 16, t2 - 50, 60, "50M", 50),               # alignment ends just before the TA
            ("r_del", 16, t3 - 52, 60, "30M2D20M", 50),            # span 52
        ])
        for primer in (DEFAULT_PRIMER, ITR + "TA"):
            counts = read_counts(self.fasta, sam, make_vars(primer))   # auto -> trimmed
            self.assertEqual(self.hit_sites(counts), {
                t1: (1, 1, 0, 0, 1, 1),
                t2: (0, 0, 1, 1, 1, 1),
                t3: (0, 0, 1, 1, 1, 1),
            }, primer)

    def test_trimmed_matches_the_historical_arithmetic_for_ungapped_reads(self):
        t1, t2 = self.sites[:2]
        sam = self.write_sam([("f", 0, t1 + 2, 60, "40M", 40), ("r", 16, t2 - 40, 60, "40M", 40)])
        counts = read_counts(self.fasta, sam, make_vars(DEFAULT_PRIMER))
        self.assertEqual(set(self.hit_sites(counts)), {t1 + 2 - 2, t2 - 40 + 40})   # POS - 2, POS + len(SEQ)

    def test_wrong_convention_misses_the_ta(self):
        t1 = self.sites[0]
        sam = self.write_sam([("f", 0, t1, 60, "50M", 50)])         # a "retained" read ...
        counts = read_counts(self.fasta, sam, make_vars(ITR, convention="trimmed"))
        self.assertNotIn(t1, self.hit_sites(counts))                # ... is credited 2 bp upstream, off the TA

    def test_tn5_arithmetic_is_unchanged(self):
        # every position is a site for Tn5; the forward site stays POS - 2 whatever the primer looks like
        sam = self.write_sam([("f", 0, 1000, 60, "50M", 50), ("r", 16, 2000, 60, "50M", 50)])
        counts = read_counts(self.fasta, sam, make_vars(ITR, transposon="Tn5", protocol="Tn5"))
        self.assertEqual(set(self.hit_sites(counts)), {998, 2050})

    def test_template_counts_both_conventions(self):
        t1, t2 = self.sites[:2]
        bc = os.path.join(self.tmp, "bc.fa")
        with open(bc, "w") as fh:
            for name in ("f", "r"):
                fh.write(">%s\nACGTACGTAC\n" % name)
        for convention, f_pos, r_pos in (("retained", t1, t2 + 1 - 49), ("trimmed", t1 + 2, t2 - 50)):
            sam = self.write_sam([("f", 99, f_pos, 60, "50M", 50), ("r", 83, r_pos, 60, "50M", 50)])
            counts = template_counts(self.fasta, sam, bc, make_vars(ITR, convention=convention))
            self.assertEqual(self.hit_sites(counts), {t1: (1, 1, 0, 0, 1, 1), t2: (0, 0, 1, 1, 1, 1)}, convention)


class TestWigSites(unittest.TestCase):
    """-wig-sites [ta|all]: which coordinates get a row in the .counts / .wig files."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(21)
        cls.genome = "".join(rng.choice("ACGT") for _ in range(3000))
        cls.sites = ta_sites(cls.genome)
        cls.tmp = tempfile.mkdtemp(prefix="tpp_wig_sites_")
        cls.fasta = os.path.join(cls.tmp, "ref.fa")
        with open(cls.fasta, "w") as fh:
            fh.write(">chr\n" + cls.genome + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def write_sam(self, records):
        path = os.path.join(self.tmp, "aln.sam")
        with open(path, "w") as fh:
            fh.write("@HD\tVN:1.6\n@SQ\tSN:chr\tLN:%d\n" % len(self.genome))
            for name, flag, pos, mapq, cigar, seq_len in records:
                fh.write("\t".join([name, str(flag), "chr", str(pos), str(mapq), cigar, "*", "0", "0", "A" * seq_len, "I" * seq_len]) + "\n")
        return path

    def off_ta_position(self, after=500):
        """A coordinate, away from the ends, whose dinucleotide is not TA."""
        return next(i + 1 for i in range(after, 2000) if self.genome[i:i + 2] != "TA")

    def test_resolution(self):
        self.assertEqual(resolve_wig_sites(make_vars(ITR)), "ta")
        self.assertEqual(resolve_wig_sites(make_vars(ITR, wig_sites="all")), "all")
        self.assertEqual(resolve_wig_sites(make_vars(ITR, wig_sites="ALL")), "all")
        # only the Himar1 (non-Mme1) tables are affected
        self.assertEqual(resolve_wig_sites(make_vars(ITR, transposon="Tn5", protocol="Tn5", wig_sites="all")), "ta")
        self.assertEqual(resolve_wig_sites(make_vars(ITR, protocol="Mme1", wig_sites="all")), "ta")
        self.assertEqual(resolve_wig_sites(Globals()), "ta")                      # objects without the attribute behave as before
        self.assertIn("every nucleotide 1..L", describe_wig_sites(make_vars(ITR, wig_sites="all")))
        self.assertIn("TA sites only", describe_wig_sites(make_vars(ITR)))

    def test_site_positions(self):
        n = len(self.genome)
        ta = [i + 1 for i in range(n - 1) if self.genome[i:i + 2] == "TA"]
        self.assertEqual(list(site_positions(self.genome, make_vars(ITR))), ta)
        self.assertEqual(list(site_positions(self.genome, make_vars(ITR, wig_sites="all"))), list(range(1, n + 1)))
        # Tn5 keeps writing 1..L-1 whatever -wig-sites says
        for choice in ("ta", "all"):
            tn5 = make_vars(ITR, transposon="Tn5", protocol="Tn5", wig_sites=choice)
            self.assertEqual(list(site_positions(self.genome, tn5)), list(range(1, n)))

    def test_option_parsing(self):
        v = Globals()
        initialize_globals(v, kwargs={"himar1": True})
        self.assertEqual(v.wig_sites, "ta")
        for choice in ("ta", "all", "All"):
            v = Globals()
            args, kwargs = clean_args(["-wig-sites", choice])
            initialize_globals(v, kwargs=kwargs)
            self.assertEqual(v.wig_sites, choice.lower())
        v = Globals()
        args, kwargs = clean_args(["-wig-sites", "everything"])
        with self.assertRaises(ValueError):
            initialize_globals(v, kwargs=kwargs)

    def test_dense_table_has_every_nucleotide_and_the_same_counts_on_ta(self):
        t1, t2 = self.sites[:2]
        off1 = self.off_ta_position()
        off2 = self.off_ta_position(after=off1 + 10)
        sam = self.write_sam([
            ("f1", 0, t1, 60, "50M", 50),
            ("f2", 0, t1, 60, "50M", 50),
            ("r1", 16, t2 + 1 - 49, 60, "50M", 50),
            ("off_f", 0, off1, 60, "50M", 50),                   # forward junction on a non-TA nucleotide
            ("off_r", 16, off2 + 1 - 49, 60, "50M", 50),         # reverse junction (end - 1) on another one
        ])
        ta_rows = read_counts(self.fasta, sam, make_vars(ITR))[0]
        all_rows = read_counts(self.fasta, sam, make_vars(ITR, wig_sites="all"))[0]
        self.assertEqual([r[0] for r in all_rows], list(range(1, len(self.genome) + 1)))
        by_pos = {r[0]: r[1:7] for r in all_rows}
        # TA sites carry exactly the counts they have in TA mode
        for row in ta_rows:
            self.assertEqual(by_pos[row[0]], row[1:7])
        # the off-TA alignments are kept at their junction coordinate
        self.assertEqual(by_pos[off1], [1, 1, 0, 0, 1, 1])
        self.assertEqual(by_pos[off2], [0, 0, 1, 1, 1, 1])
        self.assertEqual(sum(r[5] for r in ta_rows), 3)
        self.assertEqual(sum(r[5] for r in all_rows), 5)

    def test_dense_table_with_the_trimmed_convention(self):
        t1 = self.sites[0]
        off = self.off_ta_position()
        sam = self.write_sam([("f", 0, t1 + 2, 60, "50M", 50), ("off", 0, off + 2, 60, "50M", 50)])
        rows = read_counts(self.fasta, sam, make_vars(DEFAULT_PRIMER, wig_sites="all"))[0]
        hit = {r[0]: r[5] for r in rows if r[5] > 0}
        self.assertEqual(hit, {t1: 1, off: 1})

    def test_dense_template_counts(self):
        t1 = self.sites[0]
        bc = os.path.join(self.tmp, "bc.fa")
        with open(bc, "w") as fh:
            fh.write(">f\nACGTACGTAC\n")
        sam = self.write_sam([("f", 99, t1, 60, "50M", 50)])
        rows = template_counts(self.fasta, sam, bc, make_vars(ITR, wig_sites="all"))[0]
        self.assertEqual(len(rows), len(self.genome))
        self.assertEqual({r[0]: r[5] for r in rows if r[5] > 0}, {t1: 1})

    def test_mme1_and_tn5_ignore_the_option(self):
        sam = self.write_sam([("f", 0, 1000, 60, "50M", 50)])
        tn5 = read_counts(self.fasta, sam, make_vars(ITR, transposon="Tn5", protocol="Tn5", wig_sites="all"))[0]
        self.assertEqual(len(tn5), len(self.genome) - 1)
        mme1 = read_counts(self.fasta, sam, make_vars(ITR, protocol="Mme1", wig_sites="all"))[0]
        self.assertEqual([r[0] for r in mme1], [i + 1 for i in range(len(self.genome) - 1) if self.genome[i:i + 2] == "TA"])


def get_bwa():
    found = shutil.which("bwa")
    return found or ""


@unittest.skipUnless(get_bwa(), "requires BWA")
class TestEndToEnd(unittest.TestCase):
    """Run the real tpp pipeline (trimming, bwa, counting) on simulated LC113-style reads and compare to the truth."""

    @classmethod
    def setUpClass(cls):
        import pytpp.__main__
        from pytransit.globals import gui

        gui.is_active = False
        cls.tppMain = staticmethod(pytpp.__main__.main)
        rng = random.Random(11)
        cls.genome = "".join(rng.choice("ACGT") for _ in range(8000))
        cls.sites = ta_sites(cls.genome, margin=150, n=25)
        cls.tmp = tempfile.mkdtemp(prefix="tpp_e2e_")
        cls.cwd = os.getcwd()
        os.chdir(cls.tmp)  # tpp writes tpp.cfg into the working directory
        cls.fasta = os.path.join(cls.tmp, "ref.fa")
        with open(cls.fasta, "w") as fh:
            fh.write(">chr\n" + cls.genome + "\n")
        cls.truth = {}
        cls.fastq = os.path.join(cls.tmp, "reads.fastq")
        n = 0
        with open(cls.fastq, "w") as fh:
            for site in cls.sites:
                for strand in "+-":
                    for _ in range(3):
                        insert_len = 40
                        if strand == "+":
                            insert = cls.genome[site - 1:site - 1 + insert_len]          # starts with the TA
                        else:
                            insert = revcomp(cls.genome[site + 1 - insert_len:site + 1])  # starts with the TA, too
                        read = ITR + insert
                        n += 1
                        fh.write("@r%d\n%s\n+\n%s\n" % (n, read, "I" * len(read)))
                        cls.truth[site] = cls.truth.get(site, 0) + 1
        cls.n_reads = n

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_tpp(self, name, extra):
        base = os.path.join(self.tmp, name)
        args = ["-bwa", get_bwa(), "-ref", self.fasta, "-reads1", self.fastq, "-output", base, "-himar1", "-primer-start-window", "0,20"] + extra
        (a, kw) = clean_args(args)
        self.tppMain(a, kw)
        counts = {}
        with open(base + ".wig") as fh:
            for line in fh:
                if line[0] in "#v":
                    continue
                pos, n = line.split()
                if int(n) > 0:
                    counts[int(pos)] = int(n)
        with open(base + ".tn_stats") as fh:
            stats = fh.read()
        return counts, stats

    def test_primer_without_ta_is_counted_at_the_right_site(self):
        counts, stats = self.run_tpp("itr_only", ["-primer", ITR])
        self.assertEqual(counts, self.truth)
        self.assertIn("# ta_convention: retained (auto: the primer does not end in TA", stats)
        self.assertIn("# wig_sites: ta (TA sites only)", stats)

    def test_every_nucleotide_can_be_written_instead_of_the_ta_sites(self):
        base = os.path.join(self.tmp, "dense")
        args = ["-bwa", get_bwa(), "-ref", self.fasta, "-reads1", self.fastq, "-output", base, "-himar1",
                "-primer-start-window", "0,20", "-primer", ITR, "-wig-sites", "all"]
        (a, kw) = clean_args(args)
        self.tppMain(a, kw)
        coords, hit = [], {}
        with open(base + ".wig") as fh:
            for line in fh:
                if line[0] in "#v":
                    continue
                pos, n = line.split()
                coords.append(int(pos))
                if int(n) > 0:
                    hit[int(pos)] = int(n)
        self.assertEqual(coords, list(range(1, len(self.genome) + 1)))   # consecutive, 1..L
        self.assertEqual(hit, self.truth)                                  # and the same counts on the TA sites
        with open(base + ".tn_stats") as fh:
            self.assertIn("# wig_sites: all (every nucleotide 1..L", fh.read())

    def test_primer_with_ta_is_unchanged(self):
        counts, stats = self.run_tpp("itr_ta", ["-primer", ITR + "TA"])
        self.assertEqual(counts, self.truth)
        self.assertIn("# ta_convention: trimmed (auto: the primer ends in TA", stats)

    def test_default_primer_is_unchanged(self):
        counts, stats = self.run_tpp("default", [])
        self.assertEqual(counts, self.truth)
        self.assertIn("# ta_convention: trimmed (auto: the primer ends in TA", stats)

    def test_forcing_the_wrong_convention_loses_the_reads(self):
        counts, stats = self.run_tpp("forced", ["-primer", ITR, "-ta-convention", "trimmed"])
        self.assertLess(sum(counts.values()), 0.2 * self.n_reads)
        self.assertIn("# ta_convention: trimmed (set with -ta-convention)", stats)

    def test_pretrimmed_reads_need_the_explicit_option(self):
        # reads trimmed before tpp (cutadapt with the ITR only): empty primer, TA still on the read
        pre = os.path.join(self.tmp, "pretrimmed.fastq")
        with open(self.fastq) as src, open(pre, "w") as dst:
            for i, line in enumerate(src):
                dst.write(line[len(ITR):] if i % 4 in (1, 3) else line)   # drop the ITR from sequence and quality
        base = os.path.join(self.tmp, "pre")
        for extra, expect_all in (([], False), (["-ta-convention", "retained"], True)):
            args = ["-bwa", get_bwa(), "-ref", self.fasta, "-reads1", pre, "-output", base, "-himar1", "-primer", ""] + extra
            (a, kw) = clean_args(args)
            self.tppMain(a, kw)
            counts = {}
            with open(base + ".wig") as fh:
                for line in fh:
                    if line[0] in "#v":
                        continue
                    pos, n = line.split()
                    if int(n) > 0:
                        counts[int(pos)] = int(n)
            if expect_all:
                self.assertEqual(counts, self.truth)
            else:
                self.assertLess(sum(counts.values()), 0.2 * self.n_reads)


if __name__ == "__main__":
    unittest.main()
