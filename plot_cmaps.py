from dataclasses import dataclass, field

from matplotlib.colors import Colormap

from .cmap_funcs import *


@dataclass
class ColorMap:

    chem_div = get_cm("chem_div")
    chem_seq = get_cm("chem_seq")
    cryo_div = get_cm("cryo_div")
    cryo_seq = get_cm("cryo_seq")
    misc_div = get_cm("misc_div")
    misc_seq_1 = get_cm("misc_seq_1")
    misc_seq_2 = get_cm("misc_seq_2")
    misc_seq_3 = get_cm("misc_seq_3")
    misc_seq_4 = get_cm("misc_seq_4")
    ncl_div = get_cm("ncl_div")
    prec_div = get_cm("prec_div")
    prec_seq = get_cm("prec_seq")
    slev_div = get_cm("slev_div")
    slev_seq = get_cm("slev_seq")
    temp_div = get_cm("temp_div")
    temp_seq = get_cm("temp_seq")
    wind_div = get_cm("wind_div")
    wind_seq = get_cm("wind_seq")
    magma = get_cm("magma")
    inferno = get_cm("inferno")
    plasma = get_cm("plasma")
    viridis = get_cm("viridis")
    cividis = get_cm("cividis")
    twilight = get_cm("twilight")
    twilight_shifted = get_cm("twilight_shifted")
    turbo = get_cm("turbo")
    berlin = get_cm("berlin")
    managua = get_cm("managua")
    vanimo = get_cm("vanimo")
    blues = get_cm("Blues")
    brbg = get_cm("BrBG")
    bugn = get_cm("BuGn")
    bupu = get_cm("BuPu")
    cmrmap = get_cm("CMRmap")
    gnbu = get_cm("GnBu")
    greens = get_cm("Greens")
    greys = get_cm("Greys")
    orrd = get_cm("OrRd")
    oranges = get_cm("Oranges")
    prgn = get_cm("PRGn")
    piyg = get_cm("PiYG")
    pubu = get_cm("PuBu")
    pubugn = get_cm("PuBuGn")
    puor = get_cm("PuOr")
    purd = get_cm("PuRd")
    purples = get_cm("Purples")
    rdbu = get_cm("RdBu")
    rdgy = get_cm("RdGy")
    rdpu = get_cm("RdPu")
    rdylbu = get_cm("RdYlBu")
    rdylgn = get_cm("RdYlGn")
    reds = get_cm("Reds")
    spectral = get_cm("Spectral")
    wistia = get_cm("Wistia")
    ylgn = get_cm("YlGn")
    ylgnbu = get_cm("YlGnBu")
    ylorbr = get_cm("YlOrBr")
    ylorrd = get_cm("YlOrRd")
    afmhot = get_cm("afmhot")
    autumn = get_cm("autumn")
    binary = get_cm("binary")
    bone = get_cm("bone")
    brg = get_cm("brg")
    bwr = get_cm("bwr")
    cool = get_cm("cool")
    coolwarm = get_cm("coolwarm")
    copper = get_cm("copper")
    cubehelix = get_cm("cubehelix")
    flag = get_cm("flag")
    gist_earth = get_cm("gist_earth")
    gist_gray = get_cm("gist_gray")
    gist_heat = get_cm("gist_heat")
    gist_ncar = get_cm("gist_ncar")
    gist_rainbow = get_cm("gist_rainbow")
    gist_stern = get_cm("gist_stern")
    gist_yarg = get_cm("gist_yarg")
    gnuplot = get_cm("gnuplot")
    gnuplot2 = get_cm("gnuplot2")
    gray = get_cm("gray")
    hot = get_cm("hot")
    hsv = get_cm("hsv")
    jet = get_cm("jet")
    nipy_spectral = get_cm("nipy_spectral")
    ocean = get_cm("ocean")
    pink = get_cm("pink")
    prism = get_cm("prism")
    rainbow = get_cm("rainbow")
    seismic = get_cm("seismic")
    spring = get_cm("spring")
    summer = get_cm("summer")
    terrain = get_cm("terrain")
    winter = get_cm("winter")
    accent = get_cm("Accent")
    dark2 = get_cm("Dark2")
    paired = get_cm("Paired")
    pastel1 = get_cm("Pastel1")
    pastel2 = get_cm("Pastel2")
    set1 = get_cm("Set1")
    set2 = get_cm("Set2")
    set3 = get_cm("Set3")
    tab10 = get_cm("tab10")
    tab20 = get_cm("tab20")
    tab20b = get_cm("tab20b")
    tab20c = get_cm("tab20c")
    grey = get_cm("grey")
    gist_grey = get_cm("gist_grey")
    gist_yerg = get_cm("gist_yerg")
    grays = get_cm("Grays")
    thermal = get_cm("thermal")
    haline = get_cm("haline")
    solar = get_cm("solar")
    ice = get_cm("ice")
    gray = get_cm("gray")
    oxy = get_cm("oxy")
    deep = get_cm("deep")
    dense = get_cm("dense")
    algae = get_cm("algae")
    matter = get_cm("matter")
    turbid = get_cm("turbid")
    speed = get_cm("speed")
    amp = get_cm("amp")
    tempo = get_cm("tempo")
    rain = get_cm("rain")
    phase = get_cm("phase")
    topo = get_cm("topo")
    balance = get_cm("balance")
    delta = get_cm("delta")
    curl = get_cm("curl")
    diff = get_cm("diff")
    tarn = get_cm("tarn")


@dataclass
class ColorMaps:
    cm: ColorMap = field(default_factory=ColorMap)

    @staticmethod
    def new(colors: list[str], N: int = 25, *, discrete: bool = True):
        return blend(colors, N=N, discrete=discrete)

    @staticmethod
    def chem_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("chem_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def chem_seq(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("chem_seq", N, reverse, split, add_colors, discrete)

    @staticmethod
    def cryo_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("cryo_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def cryo_seq(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("cryo_seq", N, reverse, split, add_colors, discrete)

    @staticmethod
    def misc_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("misc_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def misc_seq_1(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("misc_seq_1", N, reverse, split, add_colors, discrete)

    @staticmethod
    def misc_seq_2(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("misc_seq_2", N, reverse, split, add_colors, discrete)

    @staticmethod
    def misc_seq_3(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("misc_seq_3", N, reverse, split, add_colors, discrete)

    @staticmethod
    def misc_seq_4(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("misc_seq_4", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ncl_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("ncl_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def prec_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("prec_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def prec_seq(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("prec_seq", N, reverse, split, add_colors, discrete)

    @staticmethod
    def slev_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("slev_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def slev_seq(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("slev_seq", N, reverse, split, add_colors, discrete)

    @staticmethod
    def temp_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("temp_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def temp_seq(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("temp_seq", N, reverse, split, add_colors, discrete)

    @staticmethod
    def wind_div(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("wind_div", N, reverse, split, add_colors, discrete)

    @staticmethod
    def wind_seq(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("wind_seq", N, reverse, split, add_colors, discrete)

    @staticmethod
    def magma(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("magma", N, reverse, split, add_colors, discrete)

    @staticmethod
    def inferno(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("inferno", N, reverse, split, add_colors, discrete)

    @staticmethod
    def plasma(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("plasma", N, reverse, split, add_colors, discrete)

    @staticmethod
    def viridis(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("viridis", N, reverse, split, add_colors, discrete)

    @staticmethod
    def cividis(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("cividis", N, reverse, split, add_colors, discrete)

    @staticmethod
    def twilight(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("twilight", N, reverse, split, add_colors, discrete)

    @staticmethod
    def twilight_shifted(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("twilight_shifted", N, reverse, split, add_colors, discrete)

    @staticmethod
    def turbo(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("turbo", N, reverse, split, add_colors, discrete)

    @staticmethod
    def berlin(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("berlin", N, reverse, split, add_colors, discrete)

    @staticmethod
    def managua(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("managua", N, reverse, split, add_colors, discrete)

    @staticmethod
    def vanimo(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("vanimo", N, reverse, split, add_colors, discrete)

    @staticmethod
    def blues(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Blues", N, reverse, split, add_colors, discrete)

    @staticmethod
    def brbg(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("BrBG", N, reverse, split, add_colors, discrete)

    @staticmethod
    def bugn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("BuGn", N, reverse, split, add_colors, discrete)

    @staticmethod
    def bupu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("BuPu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def cmrmap(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("CMRmap", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gnbu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("GnBu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def greens(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Greens", N, reverse, split, add_colors, discrete)

    @staticmethod
    def greys(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Greys", N, reverse, split, add_colors, discrete)

    @staticmethod
    def orrd(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("OrRd", N, reverse, split, add_colors, discrete)

    @staticmethod
    def oranges(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Oranges", N, reverse, split, add_colors, discrete)

    @staticmethod
    def prgn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("PRGn", N, reverse, split, add_colors, discrete)

    @staticmethod
    def piyg(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("PiYG", N, reverse, split, add_colors, discrete)

    @staticmethod
    def pubu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("PuBu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def pubugn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("PuBuGn", N, reverse, split, add_colors, discrete)

    @staticmethod
    def puor(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("PuOr", N, reverse, split, add_colors, discrete)

    @staticmethod
    def purd(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("PuRd", N, reverse, split, add_colors, discrete)

    @staticmethod
    def purples(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Purples", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rdbu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("RdBu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rdgy(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("RdGy", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rdpu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("RdPu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rdylbu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("RdYlBu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rdylgn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("RdYlGn", N, reverse, split, add_colors, discrete)

    @staticmethod
    def reds(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Reds", N, reverse, split, add_colors, discrete)

    @staticmethod
    def spectral(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Spectral", N, reverse, split, add_colors, discrete)

    @staticmethod
    def wistia(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Wistia", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ylgn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("YlGn", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ylgnbu(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("YlGnBu", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ylorbr(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("YlOrBr", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ylorrd(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("YlOrRd", N, reverse, split, add_colors, discrete)

    @staticmethod
    def afmhot(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("afmhot", N, reverse, split, add_colors, discrete)

    @staticmethod
    def autumn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("autumn", N, reverse, split, add_colors, discrete)

    @staticmethod
    def binary(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("binary", N, reverse, split, add_colors, discrete)

    @staticmethod
    def bone(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("bone", N, reverse, split, add_colors, discrete)

    @staticmethod
    def brg(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("brg", N, reverse, split, add_colors, discrete)

    @staticmethod
    def bwr(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("bwr", N, reverse, split, add_colors, discrete)

    @staticmethod
    def cool(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("cool", N, reverse, split, add_colors, discrete)

    @staticmethod
    def coolwarm(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("coolwarm", N, reverse, split, add_colors, discrete)

    @staticmethod
    def copper(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("copper", N, reverse, split, add_colors, discrete)

    @staticmethod
    def cubehelix(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("cubehelix", N, reverse, split, add_colors, discrete)

    @staticmethod
    def flag(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("flag", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_earth(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_earth", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_gray(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_gray", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_heat(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_heat", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_ncar(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_ncar", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_rainbow(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_rainbow", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_stern(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_stern", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_yarg(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_yarg", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gnuplot(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gnuplot", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gnuplot2(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gnuplot2", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gray(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gray", N, reverse, split, add_colors, discrete)

    @staticmethod
    def hot(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("hot", N, reverse, split, add_colors, discrete)

    @staticmethod
    def hsv(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("hsv", N, reverse, split, add_colors, discrete)

    @staticmethod
    def jet(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("jet", N, reverse, split, add_colors, discrete)

    @staticmethod
    def nipy_spectral(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("nipy_spectral", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ocean(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("ocean", N, reverse, split, add_colors, discrete)

    @staticmethod
    def pink(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("pink", N, reverse, split, add_colors, discrete)

    @staticmethod
    def prism(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("prism", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rainbow(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("rainbow", N, reverse, split, add_colors, discrete)

    @staticmethod
    def seismic(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("seismic", N, reverse, split, add_colors, discrete)

    @staticmethod
    def spring(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("spring", N, reverse, split, add_colors, discrete)

    @staticmethod
    def summer(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("summer", N, reverse, split, add_colors, discrete)

    @staticmethod
    def terrain(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("terrain", N, reverse, split, add_colors, discrete)

    @staticmethod
    def winter(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("winter", N, reverse, split, add_colors, discrete)

    @staticmethod
    def accent(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Accent", N, reverse, split, add_colors, discrete)

    @staticmethod
    def dark2(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Dark2", N, reverse, split, add_colors, discrete)

    @staticmethod
    def paired(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Paired", N, reverse, split, add_colors, discrete)

    @staticmethod
    def pastel1(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Pastel1", N, reverse, split, add_colors, discrete)

    @staticmethod
    def pastel2(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Pastel2", N, reverse, split, add_colors, discrete)

    @staticmethod
    def set1(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Set1", N, reverse, split, add_colors, discrete)

    @staticmethod
    def set2(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Set2", N, reverse, split, add_colors, discrete)

    @staticmethod
    def set3(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Set3", N, reverse, split, add_colors, discrete)

    @staticmethod
    def tab10(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("tab10", N, reverse, split, add_colors, discrete)

    @staticmethod
    def tab20(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("tab20", N, reverse, split, add_colors, discrete)

    @staticmethod
    def tab20b(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("tab20b", N, reverse, split, add_colors, discrete)

    @staticmethod
    def tab20c(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("tab20c", N, reverse, split, add_colors, discrete)

    @staticmethod
    def grey(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("grey", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_grey(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_grey", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gist_yerg(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gist_yerg", N, reverse, split, add_colors, discrete)

    @staticmethod
    def grays(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("Grays", N, reverse, split, add_colors, discrete)

    @staticmethod
    def thermal(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("thermal", N, reverse, split, add_colors, discrete)

    @staticmethod
    def haline(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("haline", N, reverse, split, add_colors, discrete)

    @staticmethod
    def solar(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("solar", N, reverse, split, add_colors, discrete)

    @staticmethod
    def ice(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("ice", N, reverse, split, add_colors, discrete)

    @staticmethod
    def gray(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("gray", N, reverse, split, add_colors, discrete)

    @staticmethod
    def oxy(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("oxy", N, reverse, split, add_colors, discrete)

    @staticmethod
    def deep(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("deep", N, reverse, split, add_colors, discrete)

    @staticmethod
    def dense(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("dense", N, reverse, split, add_colors, discrete)

    @staticmethod
    def algae(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("algae", N, reverse, split, add_colors, discrete)

    @staticmethod
    def matter(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("matter", N, reverse, split, add_colors, discrete)

    @staticmethod
    def turbid(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("turbid", N, reverse, split, add_colors, discrete)

    @staticmethod
    def speed(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("speed", N, reverse, split, add_colors, discrete)

    @staticmethod
    def amp(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("amp", N, reverse, split, add_colors, discrete)

    @staticmethod
    def tempo(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("tempo", N, reverse, split, add_colors, discrete)

    @staticmethod
    def rain(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("rain", N, reverse, split, add_colors, discrete)

    @staticmethod
    def phase(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("phase", N, reverse, split, add_colors, discrete)

    @staticmethod
    def topo(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("topo", N, reverse, split, add_colors, discrete)

    @staticmethod
    def balance(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("balance", N, reverse, split, add_colors, discrete)

    @staticmethod
    def delta(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("delta", N, reverse, split, add_colors, discrete)

    @staticmethod
    def curl(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("curl", N, reverse, split, add_colors, discrete)

    @staticmethod
    def diff(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("diff", N, reverse, split, add_colors, discrete)

    @staticmethod
    def tarn(
        N: int = 25,
        reverse: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = True,
    ) -> Colormap:
        return get_func("tarn", N, reverse, split, add_colors, discrete)


cmaps: ColorMaps = ColorMaps()
cm: ColorMaps = cmaps  # alias
