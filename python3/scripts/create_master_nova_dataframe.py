import pandas as pd
import astropy as ap
from astropy.table import Table
from astropy.coordinates import SkyCoord
import healpy as hp
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from astropy.time import Time, TimeDelta
import astropy.units as u
mpl.style.use('/home/apizzuto/Nova/python3/scripts/novae_plots.mplstyle')

###############################################################################
#######              Load all novae from Anna's paper            ##############
###############################################################################
tab = Table.read('/home/apizzuto/Nova/source_list/appendix.tex')
df = tab.to_pandas()
coords = SkyCoord(frame="galactic", l=df['$l$']*u.degree, b=df['$b$']*u.degree)
equatorial = coords.icrs
df['ra'] = equatorial.ra.deg
df['dec'] = equatorial.dec.deg
df['gamma'] = [~np.char.startswith(fl, '$<$') for fl in df['Flux']] 
df = df.replace(['-'], np.nan)
df[u'$t_2$'] = df[u'$t_2$'].astype(float)
df['Peak Time'] = [Time(t, format='iso') for t in df['Peak Time']]

###############################################################################
#######          Load gamma novae from the list we've made       ##############
###############################################################################
gamma_df = pd.read_csv('/home/apizzuto/Nova/gamma_ray_novae.csv')
gamma_coords = SkyCoord(gamma_df['RA'], gamma_df['Dec'], unit='deg')

def mjd_to_time(date):
    return Time(date, format='mjd')

gamma_df['Start Time'] = gamma_df['Start Time'].apply(mjd_to_time)
gamma_df['Stop Time'] = gamma_df['Stop Time'].apply(mjd_to_time)


###############################################################################
#######              Load novae from master galnovae list        ##############
###############################################################################
def correct_date(nov_date):
    try:
        year, month, day = nov_date.split(' ')
        try:
            day, frac_day = day.split('.')
        except:
            day = day
            frac_day = '0'
        new_time = Time(f"{year}-{month}-{day} 00:00:00", format='iso') + TimeDelta(float(f"0.{frac_day}")*86400., format='sec')
    except:
        new_time = Time(f"1800-01-01 00:00:00", format='iso')
    return new_time

def format_coords(ras, decs):
    coord_str = [f"{ra} {dec.replace('<', '')}" for ra, dec in zip(ras, decs)]
    coords = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
    new_ras = coords.ra
    new_decs = coords.dec
    return new_ras, new_decs

def clean_mag(mag):
    if mag == '':
        return None
    else:
        for substr in ['<', 'J', 'g', 'V', 'I', 'v', ':', 'R', 'B', 'p', '?', 'r']:
            mag = mag.replace(substr, '')
        if mag[-1] == '.':
            mag = mag[:-1]
        mag = float(mag)
        return mag
    
def input_name(variable, obscure):
    return np.where(variable == '', obscure, variable)

pivs = [('Name', 1), ('Date', 15), ('Variable', 32), ('RA', 44), 
        ('Dec', 60), ('Disc. Mag', 75), ('Max Mag.', 84), ('Min Mag.', 92), 
        ('T3', 104), ('Class', 111), ('Obscure xid', 119), ('Discoverer', 141), ('Refs', 182), ('', -1)]
with open('/home/apizzuto/Nova/source_list/galnovae.txt', 'r') as f:
    lines = f.readlines()
    novae = []
    for line in lines[1:-1]:
        novae.append({pivs[ii][0]: line[pivs[ii][1]-1:pivs[ii+1][1]-1].strip() for ii in range(len(pivs)-1)})

novae = pd.DataFrame(novae)

novae['Max Mag.'] = novae['Max Mag.'].apply(clean_mag)
novae['Date'] = novae['Date'].apply(correct_date)
novae = novae.drop(novae[novae['Date'] < Time("2010-01-01 00:00:00", format='iso')].index, inplace = False)
novae = novae.drop(columns=["Discoverer", "Refs", "T3", "Class"], inplace = False)
novae['RA'], novae['Dec'] = format_coords(novae['RA'], novae['Dec'])
novae['Variable'] = input_name(novae['Variable'], novae['Obscure xid'])


###############################################################################
#######              Create one master dataframe for it all      ##############
###############################################################################
master_dict = dict(Name = [], Date = [], Peak = [], RA = [], Dec = [], gamma = [],
                  gamma_start = [], gamma_stop = [], gamma_norm = [],
                  gamma_ind = [], gamma_cutoff = [])
all_names = set(novae['Variable']) | set(df['Name']) | set(gamma_df['Name'])

for name in all_names:
    if name in gamma_df['Name'].unique():
        df_ind = gamma_df[gamma_df['Name'] == name].index.values[0]
        for mkey, key in [('Name', 'Name'), ('RA', 'RA'), ('Dec', 'Dec'), 
                         ('gamma_start', 'Start Time'), ('gamma_stop', 'Stop Time'),
                         ('gamma_norm', 'Flux'), ('gamma_ind', 'Index'),
                         ('gamma_cutoff', 'Cutoff')]:
            master_dict[mkey].append(gamma_df[key][df_ind])
            
        if name in df['Name'].unique():
            df_ind = df[df['Name'] == name].index.values[0]
            print("Gamma nova found in Anna's paper, using peak from that", name)
            for mkey, key in [('Date', 'Peak Time'), ('Peak', 'Peak')]:
                master_dict[mkey].append(df[key][df_ind])
        elif name in novae['Variable'].unique():
            print("Gamma nova found in galnovae file, using peak from that", name)
            df_ind = novae[novae['Variable'] == name].index.values[0]
            for mkey, key in [('Date', 'Date'), ('Peak', 'Max Mag.')]:
                master_dict[mkey].append(novae[key][df_ind])
            
        elif name == 'V3890 Sgr':
            print('Using data from ATels for V3890 Sgr')
            master_dict['Date'].append(Time("2019-08-28 00:00:00") + TimeDelta(float(f"0.188")*86400., format='sec'))
            master_dict['Peak'].append(6.7)
        else:
            print("COULD NOT FIND GAMMA NOVA IN EITHER, NO TIME AVAILABLE?", name)
            for mkey, key in [('Date', 'Date'), ('Peak', 'Max Mag.')]:
                master_dict[mkey].append(np.nan)
        master_dict['gamma'].append(True)
        
    elif name in df['Name'].unique():
        df_ind = df[df['Name'] == name].index.values[0]
        for mkey, key in [('Name', 'Name'), ('Date', 'Peak Time'), ('Peak', 'Peak'),
                         ('RA', 'ra'), ('Dec', 'dec'), ('gamma', 'gamma')]:
            master_dict[mkey].append(df[key][df_ind])
        for mkey in ['gamma_start', 'gamma_stop', 'gamma_norm', 'gamma_ind', 'gamma_cutoff']:
            master_dict[mkey].append(np.nan)
    
    elif name in novae['Variable'].unique():
        df_ind = novae[novae['Variable'] == name].index.values[0]
        for mkey, key in [('Name', 'Variable'), ('Date', 'Date'), ('Peak', 'Max Mag.'),
                         ('RA', 'RA'), ('Dec', 'Dec')]:
            master_dict[mkey].append(novae[key][df_ind])
        master_dict['gamma'].append(False)
        for mkey in ['gamma_start', 'gamma_stop', 'gamma_norm', 'gamma_ind', 'gamma_cutoff']:
            master_dict[mkey].append(np.nan)
    else:
        print("why here tho?")
        
master_df = pd.DataFrame.from_dict(master_dict)


###############################################################################
#######         Cut out novae that occured outside of GRECO      ##############
###############################################################################
min_greco_mjd = np.min(np.load('/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/IC86_2012.data.npy')['time'])
min_greco_time = Time(min_greco_mjd, format='mjd')

max_greco_mjd = np.max(np.load('/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/IC86_2018.data.npy')['time'])
max_greco_time = Time(max_greco_mjd, format='mjd')

master_df = master_df[master_df['Date'] > min_greco_time]
master_df = master_df[master_df['Date'] < max_greco_time]

master_df = master_df.reset_index()
master_df.to_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')

###############################################################################
#######              Plot all of the novae in our dataframe      ##############
###############################################################################
fig = plt.figure(figsize=(8,4), dpi=200, facecolor='w')

gplane = SkyCoord(frame='galactic', b = np.zeros(5000)*u.degree, l = np.linspace(0.0, 360., 5000)*u.degree)
gplane_icrs = gplane.icrs
gcent = SkyCoord(frame='galactic', b = [0.0]*u.degree, l = [0.0]*u.degree)
gcent_icrs = gcent.icrs
cols = [sns.xkcd_rgb['orange pink'] if k is True else sns.xkcd_rgb['light navy blue'] for k in master_df['gamma']]
s = [14 if k is True else 10 for k in df['gamma']]

legend_els = [ Line2D([0], [0], marker='o', ls = '', color=sns.xkcd_rgb['orange pink'], label=r'$\gamma$ detected'),
              Line2D([0], [0], marker='o', ls = '', color=sns.xkcd_rgb['light navy blue'], label='Optical only')]

ax = fig.add_subplot(111, projection='mollweide')
ax.grid(True, alpha = 0.35, zorder=1, ls = '--')


equatorial = SkyCoord(ra=master_df['RA']*u.deg, dec=master_df['Dec']*u.deg)
ax.scatter(-1*equatorial.ra.wrap_at('360d').radian + np.pi, equatorial.dec.radian, 
           zorder=20, s = s, c = cols)
ax.scatter(-1*gamma_coords.ra.wrap_at('360d').radian + np.pi, gamma_coords.dec.radian, 
           zorder=20, s = s, c = sns.xkcd_rgb['orange pink'])

ax.scatter(-1.*gplane_icrs.ra.wrap_at('360d').radian + np.pi, gplane_icrs.dec.radian, 
           zorder=10, c = 'k', s = 0.5)

ax.set_xticklabels(["{:.0f}".format(v) + r'$^{\circ}$' for v in np.linspace(330., 30., 11)], fontsize = 14)
ax.set_yticklabels(["{:+.0f}".format(v) + r'$^{\circ}$' for v in np.linspace(-75., 75., 11)], fontsize = 14)
plt.text(110.*np.pi / 180., -45 * np.pi / 180, 'Equatorial\n(J2000)')
ax.legend(loc=(0.2, -0.18), handles=legend_els, ncol = 2, frameon=False)
plt.savefig('/home/apizzuto/public_html/novae/skymap_all_novae.png', dpi=200, bbox_inches='tight')