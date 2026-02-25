import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def mw():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/110, "#1a1a1a"),
    (10/110, "#405580"),
    (30/110, "#FFFFFF"),

    (40/110, "#53994d"),
    (65/110, "#e6e62e"),
    (75/110, "#ff1919"),
    (95/110, "#404040"),
    (110/110, "#000000")])
    
    return newcmp

def llmw():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/175, "#62526e"),
    (30/175, "#301830"),
    (60/175, "#48518c"),

    (70/175, "#557d97"),
    (80/175, "#7cabb9"),
    (90/175, "#FFFFFF"),

    (100/175, "#53994d"),
    (120/175, "#e6e62e"),
    (135/175, "#ff1919"),
    (145/175, "#404040"),
    (150/175, "#000000"),
    (175/175, "#FFFFFF")])

    vmin = 125
    vmax = 300
    
    return newcmp

def longTemp():
    newcmp = LinearSegmentedColormap.from_list("", [
    (00/150, "#FFFFFF"),
    (10/150, "#f2bbd4"),
    (30/150, "#4d2417"),
    (40/150, "#840f09"),
    (44/150, "#ce3f1b"),
    (47.5/150, "#e8b05f"),
    (52.5/150, "#e6d874"),
    (60/150, "#71b634"),
    (65/150, "#21843f"),
    (70/150, "#285c38"),
    (72/150, "#123c54"),
    (74/150, "#206493"),
    (76/150, "#385484"),
    (77/150, "#434c7d"),
    (82/150, "#7970f0"),
    (87.5/150, "#e7dbfe"),
    (92.5/150, "#9053e6"),
    (100/150, "#661180"),
    (105/150, "#40162e"),
    (110/150, "#bf41b9"),
    (120/150, "#FFFFFF"),
    (130/150, "#ff809f"),
    (150/150, "#ff1953")])

    return newcmp.reversed()

def tempC():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/140, "#FFFFFF"),
    (5/230, "#f2bbd4"),
    (22/140, "#4d2417"),
    (26.5/140, "#840f09"),
    (30/140, "#d41c00"),
    (32/140, "#e65005"),
    (37.5/140, "#e69018"),
    (42.5/140, "#e6d874"),
    (50/140, "#71b634"),
    (55/140, "#21843f"),
    (60/140, "#21432b"),
    (60/140, "#66b0db"),
    (64/140, "#206493"),
    (66/140, "#385484"),
    (67/140, "#434c7d"),
    (72/140, "#7970f0"),
    (77.5/140, "#e7dbfe"),
    (82.5/140, "#9053e6"),
    (90/140, "#661180"),
    (95/140, "#40162e"),
    (100/140, "#bf41b9"),
    (110/140, "#FFFFFF"),
    (120/140, "#ff809f"),
    (140/140, "#ff1953")])

    return newcmp.reversed()

def temperature():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/230, "#FFFFFF"),
    (10/230, "#f2bbd4"),
    (30/230, "#4d2417"),
    #(40/230, "#5e0027"),
    (40/230, "#840f09"),#45
    (50/230, "#ce3f1b"),
    (58/230, "#e8b05f"),
    (65/230, "#e6d874"),
    (80/230, "#71b634"),
    (90/230, "#21843f"),
    (98/230, "#21432b"),
    (98/230, "#66b0db"),
    (105/230, "#206493"),
    (109/230, "#385484"),
    (111/230, "#434c7d"),
    (120/230, "#7970f0"),
    (130/230, "#e7dbfe"),
    (140/230, "#9053e6"),
    (150/230, "#661180"),
    (160/230, "#40162e"),
    (190/230, "#bf41b9"),
    (230/230, "#ffd9ee")])

    return newcmp.reversed()

def dewp():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/120, "#eeeeee"),
    (30/120, "#2e2e2e"),
    (40/120, "#6e3e04"),
    (60/120, "#c7ad8d"),
    
    (85/120, "#1f5725"),

    (90/120, "#5c9d8d"),
    (100/120, "#385472"),
    (110/120, "#a98dc3"),
    (120/120, "#e7e7e7")])

    return newcmp

def pwat():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/80, "#eeeeee"),
    (10/80, "#2e2e2e"),
    (15/80, "#6e3e04"),
    (30/80, "#c7ad8d"),
    (40/80, "#1f5725"),
    (50/80, "#5c9d8d"),
    (60/80, "#385472"),
    (70/80, "#a98dc3"),
    (80/80, "#e7e7e7")])

    return newcmp

def rh():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (10/100, "#2e2e2e"),
    (20/100, "#6e3e04"),
    (35/100, "#c7ad8d"),

    (50/100, "#FFFFFF"),
    
    (65/100, "#5c9d8d"),
    (80/100, "#1f5725"),
    (90/100, "#1f2f40"),
    (100/100, "#100729")])

    return newcmp

def snow():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/96, "#F4F4F4"),
    (0.1/96, "#F4F4F4"),
    (0.1/96, "#BFD8EC"),
    (1/96, "#BFD8EC"),
    (1/96, "#6BAFD2"),
    (2/96, "#6BAFD2"),
    (2/96, "#2F7FBC"),
    (3/96, "#2F7FBC"),
    (3/96, "#08529B"),
    (4/96, "#08529B"),
    (4/96, "#082899"),
    (6/96, "#082899"),
    (6/96, "#FFFE94"),
    (8/96, "#FFFE94"),
    (8/96, "#FDC403"),
    (12/96, "#FDC403"),
    (12/96, "#FF8601"),
    (18/96, "#FF8601"),
    (18/96, "#D81501"),
    (24/96, "#D81501"),
    (24/96, "#980108"),
    (30/96, "#980108"),
    (30/96, "#6F0000"),
    (36/96, "#6F0000"),
    (36/96, "#320000"),
    (48/96, "#320000"),
    (48/96, "#CACBFB"),
    (60/96, "#CACBFB"),
    (60/96, "#A08BDA"),
    (72/96, "#A08BDA"),
    (72/96, "#7B509F"),
    (96/96, "#7B509F")])

    return newcmp

def pressureOld():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/100, "#FFFFFF"),
    (20/100, "#FFFFFF"),
    (20/100, "#ffbfff"),
    (50/100, "#a040ff"),
    (50/100, "#40ffff"),
    (85/100, "#204020"),
    (85/100, "#404040"),
    (100/100, "#000000")])
    
    return newcmp

def meso():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/30, "#FFFFFF"),
    (15/30, "#fc4226"),
    (30/30, "#FFFFFF")])

    return newcmp

def probs():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#FFFFFF"),
    (10/100, "#7da1a2"),
    (25/100, "#3f9349"),
    (50/100, "#e2c657"),
    (75/100, "#f7843c"),
    (90/100, "#fc4226"),
    (100/100, "#f19582")])

    return newcmp

def probs2():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (10/100, "#0c3336"),
    (25/100, "#2e6b35"),
    (50/100, "#e2c657"),
    (75/100, "#f7843c"),
    (90/100, "#fc4226"),
    (100/100, "#f19582")])

    return newcmp

def probs3():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#FFFFFF"),
    (10/100, "#7da1a2"),
    (25/100, "#3f9349"),
    (50/100, "#e2c657"),
    (75/100, "#f7843c"),
    (90/100, "#fc4226"),
    (100/100, "#b30000")])

    return newcmp

def probs4():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (20/100, "#0c3336"),
    (30/100, "#385484"),
    (40/100, "#5c9d8d"),
    (50/100, "#5e9d5c"),
    (60/100, "#e2c657"),
    (70/100, "#f7843c"),
    (80/100, "#fc4226"),
    (90/100, "#d90048"),
    (100/100, "#fd52a2")])

    return newcmp

def probs5():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (76/100, "#FFFFFF"),
    (86/100, "#0c3336"),
    (87/100, "#385484"),
    (88/100, "#5c9d8d"),
    (90/100, "#5e9d5c"),
    (92/100, "#e2c657"),
    (94/100, "#f7843c"),
    (96/100, "#fc4226"),
    (98/100, "#d90048"),
    (100/100, "#fd52a2")])

    return newcmp

def probs6():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/200, "#000000"),
    (20/200, "#0c3336"),
    (30/200, "#385484"),
    (40/200, "#5c9d8d"),
    (50/200, "#5e9d5c"),
    (60/200, "#e2c657"),
    (70/200, "#f7843c"),
    (80/200, "#fc4226"),
    (90/200, "#d90048"),
    (100/200, "#fd52a2"),
    (110/200, "#fdcae8"),
    (120/200, "#fa8de4"),
    (130/200, "#af3fb7"),
    (140/200, "#771c8e"),
    (150/200, "#5A1383"),
    (160/200, "#34095F"),
    (170/200, "#250565"),
    (180/200, "#331BA0"),
    (190/200, "#4955B1"),
    (200/200, "#A9A4CA")])

    return newcmp

def probs7():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/150, "#FFFFFF"),
    (5/150, "#fdcae8"),
    (10/150, "#fa8de4"),
    (15/150, "#af3fb7"),
    (20/150, "#771c8e"),
    (25/150, "#5A1383"),
    (30/150, "#34095F"),
    (40/150, "#1D0450"),
    (50/150, "#302C7B"),
    (60/150, "#385484"),
    (70/150, "#2d9ba3"),
    (80/150, "#336A5D"),
    (90/150, "#5e9d5c"),
    (100/150, "#e2c657"),
    (110/150, "#f7843c"),
    (120/150, "#fc4226"),
    (130/150, "#ad030e"),
    (140/150, "#680332"),
    (150/150, "#40002F")])

    return newcmp

def probs8():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/150, "#FFFFFF"),
    (5/150, "#fdcae8"),
    (10/150, "#fa8de4"),
    (15/150, "#af3fb7"),
    (20/150, "#771c8e"),
    (25/150, "#5A1383"),
    (30/150, "#34095F"),
    (40/150, "#1D0450"),
    (50/150, "#302C7B"),
    (60/150, "#385484"),
    (70/150, "#1c969e"),
    (80/150, "#59D0B4"),
    (90/150, "#89fa85"),
    (100/150, "#e2c657"),
    (110/150, "#f7843c"),
    (120/150, "#fc4226"),
    (130/150, "#ad030e"),
    (140/150, "#680332"),
    (150/150, "#40002F")])

    return newcmp

def probs9():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (5/100, "#1D0450"),
    (10/100, "#302C7B"),
    (15/100, "#385484"),
    (20/100, "#1c969e"),
    (30/100, "#59D0B4"),
    (40/100, "#89fa85"),
    (50/100, "#e2c657"),
    (60/100, "#f7843c"),
    (70/100, "#fc4226"),
    (80/100, "#ad030e"),
    (90/100, "#680332"),
    (100/100, "#40002F")])

    return newcmp

def probs10():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (10/100, "#1D0450"),
    (15/100, "#302C7B"),
    (20/100, "#385484"),
    (30/100, "#1c969e"),
    (40/100, "#4BB577"),
    (45/100, "#73d170"),
    (50/100, "#d0ce4a"),
    (55/100, "#e2c657"),
    (65/100, "#f7843c"),
    (75/100, "#e3340d"),
    (80/100, "#dd000f"),
    (90/100, "#F93573"),
    (100/100, "#FF5ED4")])

    return newcmp

def pressure():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/1000, "#FFFFFF"),
    (200/1000, "#FFFFFF"),

    (200/1000, "#204040"),
    (350/1000, "#008585"),
    (500/1000, "#74a892"),

    (500/1000, "#e5c185"),
    (700/1000, "#c7522a"),
    (850/1000, "#402820"),

    (850/1000, "#404040"),
    (1000/1000, "#000000")])
    
    return newcmp

def pressure2():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/650, "#204040"),
    (150/650, "#008585"),
    (300/650, "#74a892"),

    (300/650, "#e5c185"),
    (500/650, "#c7522a"),
    (650/650, "#402820")])
    
    return newcmp

def pressure3():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/600, "#204040"),
    (150/600, "#008585"),
    (300/600, "#74a892"),

    (300/600, "#e5c185"),
    (450/600, "#c7522a"),
    (600/600, "#402820")])
    
    return newcmp

def div():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/600, "#204040"),
    (150/600, "#008585"),
    (200/600, "#74a892"),

    (250/600, "#FFFFFF"),
    (350/600, "#FFFFFF"),

    (400/600, "#e5c185"),
    (450/600, "#c7522a"),
    (600/600, "#402820")])
    
    return newcmp

def vort():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/150, "#808080"),
    (25/150, "#FFFFFF"),
    (50/150, "#FFFFFF"),
    (70/150, "#489d45"),
    (90/150, "#e2c657"),
    (100/150, "#f7843c"),
    (115/150, "#d04900"),
    (130/150, "#a90000"),
    (150/150, "#402820")])

    return newcmp

def relh():
    newcmp = LinearSegmentedColormap.from_list("",[
    (0/100, "#000000"),
    (15/100, "#2e2e2e"),
    (25/100, "#4d2417"),
    (35/100, "#6e3e04"),
    (45/100, "#c7ad8d"),

    (50/100, "#FFFFFF"),
    
    (55/100, "#5c9d8d"),
    (65/100, "#1f5725"),
    (75/100, "#1f2f40"),
    (85/100, "#100729"),
    (100/100, "#385484")])

    return newcmp

def sst():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/32, "#730073"),
    (5/32, "#6d3a78"),
    (20/32, "#abcdff"),
    (26/32, "#fcfcff"),
    (26/32, "#fffcfc"),
    (29/32, "#e63322"),
    (32/32, "#330000")])
    
    return newcmp

def sstGFDL():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/35, "#FFFFFF"),
    (5/35, "#04132d"),
    (10/35, "#98e7dc"),
    (15/35, "#3a9229"),
    (20/35, "#ffff00"),
    (25/35, "#f25300"),
    (30/35, "#6e1103"),
    (35/35, "#000000")])

    return newcmp

def ohc():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/200, "#FFFFFF"),
    (16/200, "#04132d"),
    (60/200, "#98e7dc"),
    (80/200, "#3a9229"),
    (100/200, "#ffff00"),
    (125/200, "#f25300"),
    (150/200, "#6e1103"),
    (200/200, "#000000")])

    return newcmp

def sshws():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/140, "#5ebaff"),
    (34/140, "#5ebaff"),
    (34/140, "#00faf4"),
    (64/140, "#00faf4"),
    (64/140, "#ffffcc"),
    (83/140, "#ffffcc"),
    (83/140, "#ffe775"),
    (96/140, "#ffe775"),
    (96/140, "#ffc140"),
    (113/140, "#ffc140"),
    (113/140, "#ff8f20"),
    (137/140, "#ff8f20"),
    (137/140, "#ff6060"),
    (140/140, "#ff6060")])

    return newcmp 

def reflectivity():
    newcmp = LinearSegmentedColormap.from_list("", [ 
            (0/80, "#000000"),
            (10/80, "#000000"),
            (20/80, "#005580"),
            (30/80, "#80ff80"),
            (45/80, "#004000"),
            (50/80, "#d9d921"),
            (60/80, "#d95e21"),
            (60/80, "#d92121"),
            (70/80, "#4d1717"),
            (70/80, "#4d1732"),
            (80/80, "#e6b8cf")])

    vmin = -10
    vmax = 70
    
    return newcmp, vmin, vmax

def tempAnoms():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#f2b3b3"),
    (5/20, "#802525"),
    (7.5/20, "#faa44d"),
    (10/20, "#FFFFFF"),
    (12.5/20, "#4da4fa"),
    (15/20, "#252580"),
    (20/20, "#d3b3f2")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def tempAnoms2():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#101040"),
    (5/20, "#4040ff"),
    (9/20, "#e6e6ff"),

    (9/20, "#FFFFFF"),
    (10/20, "#FFFFFF"),
    (11/20, "#FFFFFF"),
    
    (11/20, "#ffe6e6"),
    (15/20, "#ff4040"),
    (20/20, "#401010")])

    vmin = -5
    vmax = 5

    return newcmp

def tempAnoms3():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#f2b3b3"),
    (2.5/20, "#802525"),
    (5/20, "#ff4d4d"),
    (9.5/20, "#FFFFFF"),
    (10/20, "#FFFFFF"),
    (10.5/20, "#FFFFFF"),
    (15/20, "#4d4dff"),
    (17.5/20, "#252580"),
    (20/20, "#d3b3f2")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def tempAnoms4():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#ff4d4d"),
    (10/20, "#000000"),
    (20/20, "#4d4dff")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def tempAnoms5():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#f2b3b3"),
    (5/20, "#ff4d4d"),
    (10/20, "#000000"),
    (15/20, "#4d4dff"),
    (20/20, "#d3b3f2")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def tempAnoms6():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#f2b3d2"),
    (2.5/20, "#401329"),
    (5/20, "#802525"),
    (7.5/20, "#ff4d4d"),
    (9.5/20, "#FFFFFF"),
    (10/20, "#FFFFFF"),
    (10.5/20, "#FFFFFF"),
    (12.5/20, "#4d4dff"),
    (15/20, "#252580"),
    (17.5/20, "#291340"),
    (20/20, "#d3b3f2")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def tempAnoms7():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#f2b3d2"),
    (2.5/20, "#401329"),
    (5/20, "#802525"),
    (7.5/20, "#ff4d4d"),
    (10/20, "#FFFFFF"),
    (12.5/20, "#4d4dff"),
    (15/20, "#252580"),
    (17.5/20, "#291340"),
    (20/20, "#d3b3f2")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def tempAnoms8():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/20, "#401329"),
    (5/20, "#802525"),
    (7.5/20, "#ff4d4d"),
    (10/20, "#FFFFFF"),
    (12.5/20, "#4d4dff"),
    (15/20, "#252580"),
    (20/20, "#291340")])

    vmin = -5
    vmax = 5

    return newcmp.reversed()

def stab():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/410, "#000000"),
    (100/410, "#600000"),
    (200/410, "#c50000"),
    (375/410, "#faa44d"),
    (400/410, "#FFFFFF"),
    (405/410, "#4da4fa"),
    (410/410, "#252580")])

    return newcmp.reversed()

def wind():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/160, "#000000"), 
    (34/160, "#a6a6a6"),
    (34/160, "#4245a6"),
    (64/160, "#29a668"),
    (96/160, "#cccc33"),
    (113/160, "#cc3333"),
    (137/160, "#cc7acc"),
    (160/160, "#ffffff")])
    
    vmin = 0
    vmax = 160

    return newcmp

def wind2():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/152, "#ffffff"),
    (20/152, "#74C9DA"), 
    (34/152, "#A8F39B"),
    (50/152, "#F9F797"),
    (64/152, "#f09571"),
    (83/152, "#F96B6B"),
    (96/152, "#F26EF5"),
    (113/152, "#7030A0"),
    (125/152, "#283B76"),
    (140/152, "#a28df0"),
    (152/152, "#d0cdfa")])

    return newcmp

def shear():
    newcmp = LinearSegmentedColormap.from_list("", [ 
    (0/80, "#f5f5f5"),
    (10/80, "#4245a6"),
    (15/80, "#29a668"),
    (25/80, "#cccc33"),
    (35/80, "#cc3333"),
    (50/80, "#cc7acc"),
    (80/80, "#ffffff")])

    vmin = 0
    vmax = 80
    
    return newcmp

# Shortwave Infrared
def swir():
    num1 = 15
    num2 = 55
    num3 = 40

    top = cm.get_cmap('Blues', num1)
    mid = cm.get_cmap('plasma', num2)
    bot = cm.get_cmap('bone_r', num3)

    newcolors = np.vstack((top(np.linspace(0, 1, num1)),
                           mid(np.linspace(0, 1, num2)),
                           bot(np.linspace(0, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# "Halloween" IR Colortable
def spooky():
    num1 = 20
    num2 = 85
    num3 = 30

    top = cm.get_cmap('Greys', num1)
    mid = cm.get_cmap('CMRmap', num2)
    bot = cm.get_cmap('Greys', num3)

    newcolors = np.vstack((top(np.linspace(0, 1, num1)),
                           mid(np.linspace(0, 1, num2)),
                           bot(np.linspace(0, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# "Christmas" IR Colortable
def santa():
    num1 = 20
    num2 = 30
    num3 = 50
    num4 = 40

    top = LinearSegmentedColormap.from_list("", [(0.0, "#cfcfcf"), (1, "#d9c548")])
    mid = cm.get_cmap('PiYG', num2)
    mid2 = cm.get_cmap('Reds_r', num3)
    bot = cm.get_cmap('Greys', num3)

    newcolors = np.vstack((top(np.linspace(0, 1, num1)),
                           mid(np.linspace(0.5, 1, num2)),
                           mid2(np.linspace(0.2, 0.8, num3)),
                           bot(np.linspace(0.1, 0.9, num4))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Colormap for GEOS-5 and MERRA-2 Dust Extinction Data
def dust():
    num1 = 40
    num2 = 10
    num3 = 50

    top = cm.get_cmap('Blues_r', num1)
    mid = cm.get_cmap('Greys', num2)
    bot = cm.get_cmap('afmhot', num3)

    newcolors = np.vstack((top(np.linspace(0, 1, num1)),
                           mid(np.linspace(0, 1, num2)),
                           bot(np.linspace(0, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# SSTA Colormap
def sstaOld():
    num1 = 45
    num2 = 50
    neut = 10
    num3 = 50
    num4 = 45

    top = cm.get_cmap('Reds_r', num1)
    sec = cm.get_cmap('YlOrRd', num2)
    mid = cm.get_cmap('binary', neut)
    frt = cm.get_cmap('PuBuGn_r', num3)
    bot = cm.get_cmap('BuGn', num4)

    newcolors = np.vstack((bot(np.linspace(0, 1, num1)),
                           frt(np.linspace(0, 1, num2)),
                           mid(np.linspace(0, 0.01, neut)),
                           sec(np.linspace(0, 1, num3)),
                           top(np.linspace(0, 1, num4))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Standard Deviation Colormap for GEFS Data
def stddev():
    num1 = 40
    num2 = 160

    a = cm.get_cmap('Greys_r', num1)
    b = cm.get_cmap('OrRd', num2)

    newcolors = np.vstack((a(np.linspace(0, 0.75, num1)),
                           b(np.linspace(0, 1, num2))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Three PV Colormaps for GFS and Reanalysis Data
def pv():
    top = cm.get_cmap('BuPu_r', 3)
    bottom = cm.get_cmap('OrRd', 8)

    newcolors = np.vstack((top(np.linspace(0.5, 0.75, 3)),
                        bottom(np.linspace(0, 0.5, 8))))
    newcmp = ListedColormap(newcolors, name = 'temp')
    return newcmp

def pv2():
    bottom = cm.get_cmap('OrRd', 8)

    newcolors = np.vstack((bottom(np.linspace(0, 0.5, 8))))
    newcmp = ListedColormap(newcolors, name = 'temp')
    return newcmp

def pv3():
    top = cm.get_cmap('BuPu_r', 30)
    bottom = cm.get_cmap('OrRd', 80)

    newcolors = np.vstack((top(np.linspace(0.5, 0.75, 30)),
                        bottom(np.linspace(0, 0.5, 80))))
    newcmp = ListedColormap(newcolors, name = 'temp')
    return newcmp

def pvNew():
    newcmp = LinearSegmentedColormap.from_list("", [
    (0/12, "#802525"),
    (7.5/12, "#faa44d"),
    (8/12, "#FFFFFF"),
    (10/12, "#4da4fa"),
    (12/12, "#252580")])

    return newcmp.reversed()

# Plain IR Colormap
def ir():
    color1 = 'twilight'
    num1 = 110
    color2 = 'Greys' 
    num2 = 40

    top = cm.get_cmap(color1, num1)
    bottom = cm.get_cmap(color2, num2)
    newcolors = np.vstack((top(np.linspace(0, 1, num1)),
                       bottom(np.linspace(0, 1, num2))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Older, More Detailed Colormap
def oldir():
    num1 = 20
    num2 = 60
    num3 = 40
    num4 = 20
    num5 = 10
    top3 = cm.get_cmap('OrRd', num5)
    top2 = cm.get_cmap('BuPu_r', num4)
    top = cm.get_cmap('PuRd', num1)
    mid = cm.get_cmap('Spectral', num2)
    bot = cm.get_cmap('bone_r', num3)

    newcolors = np.vstack((top3(np.linspace(0, 1, num5)), top2(np.linspace(0, 1, num4)), top(np.linspace(0, 1, num1)), mid(np.linspace(0, 1, num2)), bot(np.linspace(0, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Update to "oldir"
def whaticouldvedone():
    num1 = 25
    num2 = 55
    num3 = 40
    num4 = 10
    num5 = 10
    top3 = cm.get_cmap('OrRd', num5)
    top2 = cm.get_cmap('BuPu_r', num4)
    top = cm.get_cmap('PuRd', num1)
    mid = cm.get_cmap('Spectral', num2)
    bot = cm.get_cmap('bone_r', num3)

    newcolors = np.vstack((top3(np.linspace(0, 1, num5)), top2(np.linspace(0, 1, num4)), top(np.linspace(0, 1, num1)), mid(np.linspace(0, 1, num2)), bot(np.linspace(0, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Water Vapor Colormap
def wv():
    num1 = 20
    num2 = 40
    num3 = 30
    top = cm.get_cmap('PuRd', num1)
    mid = cm.get_cmap('BuPu_r', num2)
    bot = cm.get_cmap('Greys', num3)

    newcolors = np.vstack((top(np.linspace(0, 1, num1)), mid(np.linspace(0, 1, num2)), bot(np.linspace(0, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp

# Reflectivity Data Colormap (MRMS Data)
def ref():
    num1 = 30
    num2 = 20
    num3 = 25
    top = cm.get_cmap('summer', num1)
    mid = cm.get_cmap('autumn_r', num2)
    bot = cm.get_cmap('hot_r', num3)

    newcolors = np.vstack((top(np.linspace(0, 1, num1)), mid(np.linspace(0, 1, num2)), bot(np.linspace(0.5, 1, num3))))
    newcmp = ListedColormap(newcolors, name='temp')
    return newcmp
