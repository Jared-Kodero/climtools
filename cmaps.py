
from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .init_cmap import *

checksum = "CCC48C3FC27AA896C9B739FD8A4A254B209C2788AAC698CEE6A6ECC77D938760"



@dataclass
class Cmap:
    '''
    A utility class for working with colormaps, providing methods to create, modify, and combine colormaps in various ways.
    '''


    @staticmethod
    def create(colors: list[str], N: int = 32, *, discrete: bool = False, gamma: float = 1.0):
        ''' Create a new colormap from a list of colors '''
        return create(colors, N=N, discrete=discrete, gamma=gamma)

    @staticmethod
    def concat(
        cmap1: ListedColormap | LinearSegmentedColormap,
        cmap2: ListedColormap | LinearSegmentedColormap,
        N: int = 32,
        *,
        discrete: bool = False,
        gamma: float = 1.0
        ):
        ''' Concat two colormaps together '''
        return concat(cmap1, cmap2, N=N, discrete=discrete, gamma=gamma)

    @staticmethod
    def add(
        cmap1: ListedColormap | LinearSegmentedColormap,
        cmap2: ListedColormap | LinearSegmentedColormap,
        N: int = 32,
        *,
        discrete: bool = False,
        gamma: float = 1.0
        ):
        ''' Add two colormaps together '''
        return add_or_subtract(cmap1, cmap2, operator="+", N=N, discrete=discrete, gamma=gamma)

    @staticmethod
    def substract(
        cmap1: ListedColormap | LinearSegmentedColormap,
        cmap2: ListedColormap | LinearSegmentedColormap,
        N: int = 32,
        *,
        discrete: bool = False,
        gamma: float = 1.0
        ):
        ''' Subtract two colormaps '''
        return add_or_subtract(cmap1, cmap2, operator="-", N=N, discrete=discrete, gamma=gamma)




    @staticmethod
    def b2r_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'b2r_div' colormap '''

        return get_colormap("b2r_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def chem_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'chem_div' colormap '''

        return get_colormap("chem_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def chem_seq(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'chem_seq' colormap '''

        return get_colormap("chem_seq", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def cryo_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'cryo_div' colormap '''

        return get_colormap("cryo_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def cryo_seq(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'cryo_seq' colormap '''

        return get_colormap("cryo_seq", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def misc_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'misc_div' colormap '''

        return get_colormap("misc_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def misc_seq_1(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'misc_seq_1' colormap '''

        return get_colormap("misc_seq_1", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def misc_seq_2(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'misc_seq_2' colormap '''

        return get_colormap("misc_seq_2", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def misc_seq_3(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'misc_seq_3' colormap '''

        return get_colormap("misc_seq_3", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def misc_seq_4(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'misc_seq_4' colormap '''

        return get_colormap("misc_seq_4", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ncl_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'ncl_div' colormap '''

        return get_colormap("ncl_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def prec_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'prec_div' colormap '''

        return get_colormap("prec_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def prec_seq(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'prec_seq' colormap '''

        return get_colormap("prec_seq", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def slev_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'slev_div' colormap '''

        return get_colormap("slev_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def slev_seq(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'slev_seq' colormap '''

        return get_colormap("slev_seq", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def soilmoist_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'soilmoist_div' colormap '''

        return get_colormap("soilmoist_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def temp_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'temp_div' colormap '''

        return get_colormap("temp_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def temp_seq(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'temp_seq' colormap '''

        return get_colormap("temp_seq", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def wind_div(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'wind_div' colormap '''

        return get_colormap("wind_div", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def wind_seq(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'wind_seq' colormap '''

        return get_colormap("wind_seq", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def magma(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'magma' colormap '''

        return get_colormap("magma", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def inferno(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'inferno' colormap '''

        return get_colormap("inferno", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def plasma(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'plasma' colormap '''

        return get_colormap("plasma", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def viridis(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'viridis' colormap '''

        return get_colormap("viridis", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def cividis(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'cividis' colormap '''

        return get_colormap("cividis", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def twilight(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'twilight' colormap '''

        return get_colormap("twilight", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def twilight_shifted(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'twilight_shifted' colormap '''

        return get_colormap("twilight_shifted", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def turbo(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'turbo' colormap '''

        return get_colormap("turbo", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def berlin(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'berlin' colormap '''

        return get_colormap("berlin", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def managua(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'managua' colormap '''

        return get_colormap("managua", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def vanimo(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'vanimo' colormap '''

        return get_colormap("vanimo", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def blues(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Blues' colormap '''

        return get_colormap("Blues", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def brbg(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'BrBG' colormap '''

        return get_colormap("BrBG", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def bugn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'BuGn' colormap '''

        return get_colormap("BuGn", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def bupu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'BuPu' colormap '''

        return get_colormap("BuPu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def cmrmap(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'CMRmap' colormap '''

        return get_colormap("CMRmap", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gnbu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'GnBu' colormap '''

        return get_colormap("GnBu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def greens(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Greens' colormap '''

        return get_colormap("Greens", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def greys(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Greys' colormap '''

        return get_colormap("Greys", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def orrd(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'OrRd' colormap '''

        return get_colormap("OrRd", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def oranges(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Oranges' colormap '''

        return get_colormap("Oranges", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def prgn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'PRGn' colormap '''

        return get_colormap("PRGn", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def piyg(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'PiYG' colormap '''

        return get_colormap("PiYG", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def pubu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'PuBu' colormap '''

        return get_colormap("PuBu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def pubugn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'PuBuGn' colormap '''

        return get_colormap("PuBuGn", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def puor(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'PuOr' colormap '''

        return get_colormap("PuOr", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def purd(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'PuRd' colormap '''

        return get_colormap("PuRd", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def purples(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Purples' colormap '''

        return get_colormap("Purples", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rdbu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'RdBu' colormap '''

        return get_colormap("RdBu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rdgy(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'RdGy' colormap '''

        return get_colormap("RdGy", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rdpu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'RdPu' colormap '''

        return get_colormap("RdPu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rdylbu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'RdYlBu' colormap '''

        return get_colormap("RdYlBu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rdylgn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'RdYlGn' colormap '''

        return get_colormap("RdYlGn", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def reds(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Reds' colormap '''

        return get_colormap("Reds", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def spectral(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Spectral' colormap '''

        return get_colormap("Spectral", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def wistia(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Wistia' colormap '''

        return get_colormap("Wistia", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ylgn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'YlGn' colormap '''

        return get_colormap("YlGn", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ylgnbu(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'YlGnBu' colormap '''

        return get_colormap("YlGnBu", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ylorbr(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'YlOrBr' colormap '''

        return get_colormap("YlOrBr", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ylorrd(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'YlOrRd' colormap '''

        return get_colormap("YlOrRd", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def afmhot(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'afmhot' colormap '''

        return get_colormap("afmhot", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def autumn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'autumn' colormap '''

        return get_colormap("autumn", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def binary(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'binary' colormap '''

        return get_colormap("binary", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def bone(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'bone' colormap '''

        return get_colormap("bone", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def brg(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'brg' colormap '''

        return get_colormap("brg", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def bwr(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'bwr' colormap '''

        return get_colormap("bwr", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def cool(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'cool' colormap '''

        return get_colormap("cool", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def coolwarm(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'coolwarm' colormap '''

        return get_colormap("coolwarm", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def copper(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'copper' colormap '''

        return get_colormap("copper", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def cubehelix(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'cubehelix' colormap '''

        return get_colormap("cubehelix", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def flag(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'flag' colormap '''

        return get_colormap("flag", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_earth(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_earth' colormap '''

        return get_colormap("gist_earth", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_gray(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_gray' colormap '''

        return get_colormap("gist_gray", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_heat(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_heat' colormap '''

        return get_colormap("gist_heat", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_ncar(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_ncar' colormap '''

        return get_colormap("gist_ncar", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_rainbow(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_rainbow' colormap '''

        return get_colormap("gist_rainbow", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_stern(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_stern' colormap '''

        return get_colormap("gist_stern", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_yarg(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_yarg' colormap '''

        return get_colormap("gist_yarg", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gnuplot(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gnuplot' colormap '''

        return get_colormap("gnuplot", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gnuplot2(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gnuplot2' colormap '''

        return get_colormap("gnuplot2", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gray(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gray' colormap '''

        return get_colormap("gray", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def hot(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'hot' colormap '''

        return get_colormap("hot", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def hsv(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'hsv' colormap '''

        return get_colormap("hsv", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def jet(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'jet' colormap '''

        return get_colormap("jet", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def nipy_spectral(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'nipy_spectral' colormap '''

        return get_colormap("nipy_spectral", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ocean(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'ocean' colormap '''

        return get_colormap("ocean", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def pink(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'pink' colormap '''

        return get_colormap("pink", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def prism(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'prism' colormap '''

        return get_colormap("prism", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rainbow(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'rainbow' colormap '''

        return get_colormap("rainbow", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def seismic(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'seismic' colormap '''

        return get_colormap("seismic", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def spring(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'spring' colormap '''

        return get_colormap("spring", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def summer(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'summer' colormap '''

        return get_colormap("summer", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def terrain(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'terrain' colormap '''

        return get_colormap("terrain", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def winter(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'winter' colormap '''

        return get_colormap("winter", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def accent(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Accent' colormap '''

        return get_colormap("Accent", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def dark2(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Dark2' colormap '''

        return get_colormap("Dark2", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def paired(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Paired' colormap '''

        return get_colormap("Paired", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def pastel1(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Pastel1' colormap '''

        return get_colormap("Pastel1", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def pastel2(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Pastel2' colormap '''

        return get_colormap("Pastel2", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def set1(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Set1' colormap '''

        return get_colormap("Set1", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def set2(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Set2' colormap '''

        return get_colormap("Set2", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def set3(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Set3' colormap '''

        return get_colormap("Set3", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def tab10(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'tab10' colormap '''

        return get_colormap("tab10", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def tab20(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'tab20' colormap '''

        return get_colormap("tab20", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def tab20b(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'tab20b' colormap '''

        return get_colormap("tab20b", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def tab20c(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'tab20c' colormap '''

        return get_colormap("tab20c", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def grey(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'grey' colormap '''

        return get_colormap("grey", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_grey(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_grey' colormap '''

        return get_colormap("gist_grey", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def gist_yerg(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'gist_yerg' colormap '''

        return get_colormap("gist_yerg", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def grays(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'Grays' colormap '''

        return get_colormap("Grays", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def thermal(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'thermal' colormap '''

        return get_colormap("thermal", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def haline(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'haline' colormap '''

        return get_colormap("haline", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def solar(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'solar' colormap '''

        return get_colormap("solar", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def ice(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'ice' colormap '''

        return get_colormap("ice", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def oxy(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'oxy' colormap '''

        return get_colormap("oxy", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def deep(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'deep' colormap '''

        return get_colormap("deep", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def dense(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'dense' colormap '''

        return get_colormap("dense", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def algae(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'algae' colormap '''

        return get_colormap("algae", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def matter(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'matter' colormap '''

        return get_colormap("matter", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def turbid(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'turbid' colormap '''

        return get_colormap("turbid", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def speed(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'speed' colormap '''

        return get_colormap("speed", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def amp(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'amp' colormap '''

        return get_colormap("amp", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def tempo(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'tempo' colormap '''

        return get_colormap("tempo", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def rain(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'rain' colormap '''

        return get_colormap("rain", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def phase(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'phase' colormap '''

        return get_colormap("phase", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def topo(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'topo' colormap '''

        return get_colormap("topo", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def balance(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'balance' colormap '''

        return get_colormap("balance", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def delta(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'delta' colormap '''

        return get_colormap("delta", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def curl(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'curl' colormap '''

        return get_colormap("curl", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def diff(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'diff' colormap '''

        return get_colormap("diff", N, r, split, add_colors, discrete,as_colors, gamma)




    @staticmethod
    def tarn(
        N: int = 32,
        r: bool = False,
        split: tuple[float, float] = (0, 1),
        add_colors: dict[int, str | list[str]] = None,
        discrete: bool = False,
        as_colors : bool = False,
        gamma: float = 1.0
    ) -> ListedColormap | LinearSegmentedColormap:
        ''' Get the 'tarn' colormap '''

        return get_colormap("tarn", N, r, split, add_colors, discrete,as_colors, gamma)


