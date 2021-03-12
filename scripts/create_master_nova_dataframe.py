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
mpl.style.use('/home/apizzuto/Nova/scripts/novae_plots.mplstyle')

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
#######   Load novae from x-ray/gamma analysis (used for gamma times)      ####
###############################################################################
xray_tab = Table.read('/home/apizzuto/Nova/source_list/gamma_nova_from_xray_paper.tex').to_pandas()

tstart = []
for t in xray_tab['Time$_{\\rm \\gamma-ray\\ start}$']:
    if 'table' in t:
        t = t.split('\\')[0]
    tmp = Time(t, format='mjd')
    tstart.append(tmp)
    
tstop = []
for t in xray_tab['Time$_{\\rm \\gamma-ray\\ end}$']:
    if 'table' in t:
        t = t.split('\\')[0]
        t = t.split('--')[0]
    tmp = Time(t, format='mjd')
    tstop.append(tmp)

xray_tab['Time$_{\\rm \\gamma-ray\\ start}$'] = tstart
xray_tab['Time$_{\\rm \\gamma-ray\\ end}$'] = tstop

###############################################################################
#######             Load novae from optical peak time stuff              ######
###############################################################################
optical_info = pd.read_csv('/home/apizzuto/Nova/source_list/nova_optical_peaks.csv')
optical_info['OpticalPeak'] = [Time(t, format='iso') for t in optical_info['OpticalPeak']]

###############################################################################
#######             Keep dictionary of the extra peak mags              ######
###############################################################################
peak_mag = {
    'V3661 Oph': 10.6, 'V3662 Oph': 14.1, 'V659 Sct': 8.3,
    'V1659 Sco': 12.3, 'V1656 Sco': 11.4, 'V1707 Sco': 10.3,
    'V5853 Sgr': 10.7, 'V2860 Ori': 9.4, 'V611 Sct': 13.4,
    'V5856 Sgr': 5.4, 'V670 Ser': 11.8, 'V1660 Sco': 13.0,
    'V549 Vel': 9.3, 'V392 Per': 6.2, 'V555 Nor': 12.43,
    'MGAB-V207': 3.7
}

###############################################################################
#######              Create one master dataframe for it all      ##############
###############################################################################
master_dict = dict(Name = [], Date = [], Peak = [], RA = [], Dec = [], gamma = [],
                  gamma_start = [], gamma_stop = [], gamma_norm = [],
                  gamma_ind = [], gamma_cutoff = [],
                  refs = [], )
all_names = set(novae['Variable']) | set(df['Name']) | set(gamma_df['Name'])

for name in all_names:
    print(f"Beginning {name}")
    if name in ["V1404 Cen", "V5854 Sgr", "V1657 Sco", "V3663 Oph", 
        "FM Cir", "V3731 Oph", "V3730 Oph", "V2891 Cyg", "V1709 Sco"]:
        print(f"Skipping {name}")
        continue
    master_dict['refs'].append('')
    if name in gamma_df['Name'].unique():
        print(f"\t - is gamma nova")
        df_ind = gamma_df[gamma_df['Name'] == name].index.values[0]
        for mkey, key in [('Name', 'Name'), ('RA', 'RA'), ('Dec', 'Dec'), 
                         # ('gamma_start', 'Start Time'), ('gamma_stop', 'Stop Time'),
                         ('gamma_norm', 'Flux'), ('gamma_ind', 'Index'),
                         ('gamma_cutoff', 'Cutoff')]:
            master_dict[mkey].append(gamma_df[key][df_ind])
        if name in xray_tab['Nova'].unique():
            print(f"\t - gamma nova {name} included in x-ray analysis,")
            print("\t\t using gamma time from that paper")
            xind = xray_tab[xray_tab['Nova'] == name].index.values[0]
            for mkey, x_key, key in [('gamma_start', 'Time$_{\\rm \\gamma-ray\\ start}$', 'Start Time'), 
                ('gamma_stop', 'Time$_{\\rm \\gamma-ray\\ end}$', 'Stop Time')]:
                master_dict[mkey].append(xray_tab[x_key][xind])
            master_dict['refs'][-1] = master_dict['refs'][-1] + 'xray:' + xray_tab['Reference'][xind]
        else:
            print(f"\t - Gamma nova {name} not included in x-ray analysis")
            print("\t\t using gamma time from our own list")
            for mkey, x_key, key in [('gamma_start', 'Time$_{\\rm \\gamma-ray\\ start}$', 'Start Time'), 
                ('gamma_stop', 'Time$_{\\rm \\gamma-ray\\ end}$', 'Stop Time')]:
                master_dict[mkey].append(gamma_df[key][df_ind])
            
        if name in df['Name'].unique():
            df_ind = df[df['Name'] == name].index.values[0]
            print("\t - Gamma nova found in Anna's paper, using peak from that", name)
            for mkey, key in [('Date', 'Peak Time'), ('Peak', 'Peak')]:
                master_dict[mkey].append(df[key][df_ind])
            master_dict['refs'][-1] = master_dict['refs'][-1] + 'Anna'  
        elif name in optical_info['Name'].unique():
            opt_ind = optical_info[optical_info['Name'] == name].index.values[0]
            print(f"\t - Gamma nova {name} found in our list of optical times")
            print("\t\t using time from that list")
            master_dict['Date'].append(optical_info['OpticalPeak'][opt_ind])
            master_dict['refs'][-1] = master_dict['refs'][-1] + 'optical:' + optical_info['Ref'][opt_ind]
            if name in novae['Variable'].unique():
                print("\t - Taking peak mag from galnovae file for", name)
                df_ind = novae[novae['Variable'] == name].index.values[0]
                for mkey, key in [('Peak', 'Max Mag.')]:
                    master_dict[mkey].append(novae[key][df_ind]) 
            else:
                if name == 'V3890 Sgr':
                    print('\t - Using data from ATels for V3890 Sgr')
                    master_dict['Peak'].append(6.7)
                    master_dict['refs'][-1] = master_dict['refs'][-1] + 'optical:Atel13047'
                else:
                    print(f"\t - NO MAX MAGNITUDE FOR NOVA {name}")
        else:
            print("\t - COULD NOT FIND GAMMA NOVA IN EITHER, NO TIME AVAILABLE?", name)
            for mkey, key in [('Date', 'Date')]:
                master_dict[mkey].append(np.nan)
            master_dict['Peak'].append(peak_mag[name])
        master_dict['gamma'].append(True)
        
    elif name in df['Name'].unique():
        print(f"\t - Found in Anna's paper, all info from there")
        df_ind = df[df['Name'] == name].index.values[0]
        for mkey, key in [('Name', 'Name'), ('Date', 'Peak Time'), ('Peak', 'Peak'),
                         ('RA', 'ra'), ('Dec', 'dec'), ('gamma', 'gamma')]:
            master_dict[mkey].append(df[key][df_ind])
        for mkey in ['gamma_start', 'gamma_stop', 'gamma_norm', 'gamma_ind', 'gamma_cutoff']:
            master_dict[mkey].append(np.nan)
        master_dict['refs'][-1] = master_dict['refs'][-1] + 'Anna'
    
    elif name in novae['Variable'].unique():
        print(f"\t - Found in Galnovae file, getting info from there")
        df_ind = novae[novae['Variable'] == name].index.values[0]
        for mkey, key in [('Name', 'Variable'), ('Peak', 'Max Mag.'),
                         ('RA', 'RA'), ('Dec', 'Dec')]:
            master_dict[mkey].append(novae[key][df_ind])
        if name in optical_info['Name'].unique():
            print(f"\t - Optical peak time from local file")
            opt_ind = optical_info[optical_info['Name'] == name].index.values[0]
            for mkey, opt_key in [('Date', 'OpticalPeak')]:
                master_dict[mkey].append(optical_info[opt_key][opt_ind])
            master_dict['refs'][-1] = master_dict['refs'][-1] + 'optical: ' + optical_info['Ref'][opt_ind]
        else:
            print(f"\t - No info on optical peak for {name}")
            master_dict['Date'].append(np.nan)
        master_dict['gamma'].append(False)
        for mkey in ['gamma_start', 'gamma_stop', 'gamma_norm', 'gamma_ind', 'gamma_cutoff']:
            master_dict[mkey].append(np.nan)
    else:
        print("why here tho?")

for i, (name, p) in enumerate(zip(master_dict['Name'], master_dict['Peak'])):
    try:
        if np.isnan(p):
            master_dict['Peak'][i] = peak_mag[name]
    except:
        pass
        
master_df = pd.DataFrame.from_dict(master_dict)


###############################################################################
#######         Cut out novae that occured outside of GRECO      ##############
###############################################################################
min_greco_mjd = np.min(np.load('/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/IC86_2012.data.npy')['time'])
min_greco_time = Time(min_greco_mjd, format='mjd')

max_greco_mjd = np.max(np.load('/data/user/apizzuto/Nova/GRECO_Skylab_Dataset/v2.4/IC86_2018.data.npy')['time'])
max_greco_mjd = 59000. # End of May 2020 lines up with end of 2019 data
max_greco_time = Time(max_greco_mjd, format='mjd')

master_df = master_df[master_df['Date'] > min_greco_time]
master_df = master_df[master_df['Date'] < max_greco_time]

master_df = master_df.reset_index()

# pd.set_option('display.max_rows', 100)
# pd.options.display.max_colwidth = 100
# print(master_df)
# import sys
# sys.exit()

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
s = [14 if k is True else 10 for k in master_df['gamma']]

legend_els = [ Line2D([0], [0], marker='o', ls = '', color=sns.xkcd_rgb['orange pink'], label=r'$\gamma$ detected'),
              Line2D([0], [0], marker='o', ls = '', color=sns.xkcd_rgb['light navy blue'], label='Optical only')]

ax = fig.add_subplot(111, projection='mollweide')
ax.grid(True, alpha = 0.35, zorder=1, ls = '--')


equatorial = SkyCoord(ra=master_df['RA']*u.deg, dec=master_df['Dec']*u.deg)
ax.scatter(-1*equatorial.ra.wrap_at('360d').radian + np.pi, equatorial.dec.radian, 
           zorder=20, s = s, c = cols)
# ax.scatter(-1*gamma_coords.ra.wrap_at('360d').radian + np.pi, gamma_coords.dec.radian, 
#            zorder=20, s = s, c = sns.xkcd_rgb['orange pink'])

ax.scatter(-1.*gplane_icrs.ra.wrap_at('360d').radian + np.pi, gplane_icrs.dec.radian, 
           zorder=10, c = 'k', s = 0.5)

ax.set_xticklabels(["{:.0f}".format(v) + r'$^{\circ}$' for v in np.linspace(330., 30., 11)], fontsize = 14)
ax.set_yticklabels(["{:+.0f}".format(v) + r'$^{\circ}$' for v in np.linspace(-75., 75., 11)], fontsize = 14)
plt.text(110.*np.pi / 180., -45 * np.pi / 180, 'Equatorial\n(J2000)')
ax.legend(loc=(0.2, -0.18), handles=legend_els, ncol = 2, frameon=False)
plt.savefig('/home/apizzuto/public_html/novae/skymap_all_novae.png', dpi=200, bbox_inches='tight')


###############################################################################
#######              Make latex table of novae for paper      ##############
###############################################################################
def clean_date(date):
    return date.iso.split(' ')[0]

def clean_ra(ra):
    return f"{ra:.2f}" + "$^{\circ}$"

def clean_dec(dec):
    return f"{dec:+.2f}" + "$^{\circ}$"

def clean_gam(gamma):
    return '\checkmark' if gamma else '--'

novae = pd.read_pickle('/home/apizzuto/Nova/master_nova_dataframe.pkl')
novae = novae.sort_values('Date')
novae = novae[["Name", "Date", "RA", "Dec", "Peak", "gamma", "refs"]]
for col, cleaner in [('Date', clean_date), ('RA', clean_ra), ('Dec', clean_dec),
                    ('gamma', clean_gam)]:
    novae[col] = novae[col].apply(cleaner)
    
novae = novae.rename(columns={'Date': 'Peak Date', 'RA': '$\\alpha$',
                             'Dec': '$\delta$', 'gamma': '$\gamma$-detected',
                             'refs': 'Reference', 'Peak': "Peak Mag."})
    
latex_str = novae.to_latex(longtable=True, index_names=False, index=False)
latex_str = latex_str.replace('\\textasciicircum ', '^')
latex_str = latex_str.replace('\$', '$')
latex_str = latex_str.replace('\\textbackslash ', '\\')
latex_str = latex_str.replace('\\{', '{')
latex_str = latex_str.replace('\\}', '}')
with open('./tmp_latex_table_paper.tex', 'w') as fi:
    fi.writelines(latex_str)
