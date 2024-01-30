import numpy as np
import datetime
from datetime import date, timedelta
from scipy.spatial import KDTree
import xarray as xr


def MITprof_from_fields(prof_depth, prof_descr, prof_YYYYMMDD,
                prof_HHMMSS, prof_lon, prof_lat,
                prof_point, prof_T, prof_Tweight,
                prof_Terr, prof_S, prof_Sweight,
                prof_Serr, prof_interp_XC11, prof_interp_YC11,
                prof_interp_XCNINJ, prof_interp_YCNINJ, prof_interp_i,
                prof_interp_j, prof_interp_lon, prof_interp_lat):

    mitprof = MITprof(prof_depth, prof_descr, prof_YYYYMMDD,
                prof_HHMMSS, prof_lon, prof_lat,
                prof_point, prof_T, prof_Tweight,
                prof_Terr, prof_S, prof_Sweight,
                prof_Serr, prof_interp_XC11, prof_interp_YC11,
                prof_interp_XCNINJ, prof_interp_YCNINJ, prof_interp_i,
                prof_interp_j, prof_interp_lon, prof_interp_lat)
    return mitprof.assemble_dataset()

def MITprof_from_metadata(grid_ds,
                  start_date, end_date, freq,
                  ungridded_lat, ungridded_lon,
                  sNx, sNy, fld_data_dict=None):

    # load grid dict
    xc = grid_ds.XC.values
    yc = grid_ds.YC.values
    prof_depth = abs(grid_ds.Z.values)
    nz = len(prof_depth)

    # get gridded lat/lon, prof_point
    latlon_dict = get_nearest_latlon(xc, yc, ungridded_lat, ungridded_lon)
    prof_point = latlon_dict['prof_point']
    ns = len(prof_point)
   
    # get temporal dict
    time_dict, nt = get_time_dict(ns, start_date, end_date, freq)
    
    # fill data fields
    flds_dict, fld_strs, fld_exts = get_fields_dict(ns, nt, nz, fld_data_dict)

    # get tile data
    tile_dict = get_tile_dict(xc, yc, prof_point, sNx, sNy)

    # compile all dicts into prof_dict
    prof_dict = get_prof_dict(latlon_dict, time_dict, flds_dict, tile_dict)
    prof_dict['prof_depth'] = prof_depth

    return MITprof(fld_strs=fld_strs, fld_exts=fld_exts, **prof_dict).assemble_dataset()


class MITprof:
    def __init__(self,
            prof_depth=None, prof_descr=None, prof_YYYYMMDD=None,
            prof_HHMMSS=None, prof_lon=None, prof_lat=None,
            prof_point=None, prof_T=None, prof_Tweight=None,
            prof_Terr=None, prof_S=None, prof_Sweight=None,
            prof_Serr=None, prof_interp_XC11=None, prof_interp_YC11=None,
            prof_interp_XCNINJ=None, prof_interp_YCNINJ=None, prof_interp_i=None,
            prof_interp_j=None, prof_interp_lon=None, prof_interp_lat=None,
            prof_interp_weights=None, fld_strs=None, fld_exts=None,
        ):
    
        self.prof_depth=prof_depth
        self.prof_descr=prof_descr
        self.prof_YYYYMMDD=prof_YYYYMMDD
        self.prof_HHMMSS=prof_HHMMSS
        self.prof_lon=prof_lon
        self.prof_lat=prof_lat
        self.prof_point=prof_point
        self.prof_T=prof_T
        self.prof_Tweight=prof_Tweight
        self.prof_Terr=prof_Terr
        self.prof_S=prof_S
        self.prof_Sweight=prof_Sweight
        self.prof_Serr=prof_Serr
        self.prof_interp_XC11=prof_interp_XC11
        self.prof_interp_YC11=prof_interp_YC11
        self.prof_interp_XCNINJ=prof_interp_XCNINJ
        self.prof_interp_YCNINJ=prof_interp_YCNINJ
        self.prof_interp_i=prof_interp_i
        self.prof_interp_j=prof_interp_j
        self.prof_interp_lon=prof_interp_lon
        self.prof_interp_lat=prof_interp_lat
        self.prof_interp_weights=prof_interp_weights
        self.fld_strs=fld_strs
        self.fld_exts = fld_exts

    def assemble_dataset(self):
        mitprof = xr.Dataset(
                data_vars=dict(
                    prof_depth=(['iDEPTH'],self.prof_depth),
                    prof_descr=(['iPROF'],self.prof_descr),
                    prof_YYYYMMDD=(['iPROF'],self.prof_YYYYMMDD),
                    prof_HHMMSS=(['iPROF'],self.prof_HHMMSS),
                    prof_lon=(['iPROF'],self.prof_lon),
                    prof_lat=(['iPROF'],self.prof_lat),
                    prof_point=(['iPROF'],self.prof_point),
                    prof_interp_XC11=(['iPROF'],self.prof_interp_XC11),
                    prof_interp_YC11=(['iPROF'],self.prof_interp_YC11),
                    prof_interp_XCNINJ=(['iPROF'],self.prof_interp_XCNINJ),
                    prof_interp_YCNINJ=(['iPROF'],self.prof_interp_YCNINJ),
                    prof_interp_i=(['iPROF', 'iINTERP'],self.prof_interp_i[:, None]),
                    prof_interp_j=(['iPROF', 'iINTERP'],self.prof_interp_j[:, None]),
                    prof_interp_lon=(['iPROF', 'iINTERP'],self.prof_interp_lon[:, None]),
                    prof_interp_lat=(['iPROF', 'iINTERP'],self.prof_interp_lat[:, None]),
                    prof_interp_weights=(['iPROF', 'iINTERP'],self.prof_interp_weights[:, None]),
                )
        )

        # populate observation fields
        for fld in self.fld_strs:
            for fld_ext in self.fld_exts:
                fld_tmp = getattr(self, fld+fld_ext)
                mitprof[fld+fld_ext] = (['iPROF', 'iDEPTH'], fld_tmp)

        return mitprof


def get_nearest_latlon(xc, yc, prof_lat, prof_lon, get_unique=True, verbose=False):
    # get gridded/llc coordinates

    llc_coords = np.c_[yc.ravel(), xc.ravel()]
    sensor_coords = np.c_[prof_lat, prof_lon]

    kd_tree = KDTree(llc_coords)
    distance, nearest_grid_idx = kd_tree.query(sensor_coords, k=1)

    assert((nearest_grid_idx>np.prod(xc.shape)).sum()==0)

    prof_lat_out = prof_lat
    prof_lon_out = prof_lon

    nearest_grid_idx_uniq, index = np.unique(nearest_grid_idx, return_index=True)

    prof_interp_lat = yc.ravel()[nearest_grid_idx]
    prof_interp_lon = xc.ravel()[nearest_grid_idx]

    latlon_dict = dict()

    if get_unique:
        if len(nearest_grid_idx) > len(nearest_grid_idx_uniq):
            print("Warning: Mapping from ungridded to gridded was not one-to-one!")
            if verbose:

                for i, (lat0, lon0, lat1, lon1) in enumerate(zip(prof_lat, prof_lon,
                                    prof_interp_lat, prof_interp_lon)):
                    print("({:.2f},{:.2f}) -> ({:.2f},{:.2f})".format(lat0, lon0, lat1, lon1))
            print("Returning nearest UNIQUE (lat,lon) pairs")
            print("{} ungridded (lat,lon) pairs -> {} gridded (lat,lon) pairs".format(len(prof_lat),len(nearest_grid_idx_uniq)))

            prof_lat_out = np.array(prof_lat)[index]
            prof_lon_out = np.array(prof_lon)[index]
            prof_interp_lat = prof_interp_lat[index]
            prof_interp_lon = prof_interp_lon[index]
            distance_uniq = distance[index]

        latlon_dict['prof_point']=nearest_grid_idx_uniq
    else:
        latlon_dict['prof_point']=nearest_grid_idx


    latlon_dict['prof_lat']=prof_lat_out
    latlon_dict['prof_lon']=prof_lon_out
    latlon_dict['prof_interp_lat']=prof_interp_lat
    latlon_dict['prof_interp_lon']=prof_interp_lon
    latlon_dict['prof_interp_weights']=np.ones_like(prof_lat_out)

    return latlon_dict

def get_time_dict(ns, start_date, end_date, freq='daily'):

    supported_freqs = ['daily', 'hourly']
    delta = end_date - start_date
    n_yyyymmdd = delta.days + 1

    if freq == 'hourly':
        # this assumes user wants data right on the hour
        prof_HHMMSS = np.arange(0,24e4,1e4).astype(int)
        fac_yyyymmdd = 24
    elif freq == 'daily':
        # this assumes user wants data on first hour of each day
        prof_HHMMSS = 0
        fac_yyyymmdd = 1
    else: 
        raise Exception('invalid frequency \'{}\'\nCurrently supported frequencies: {}'.\
                        format(freq, '\'' + '\', \''.join(supported_freqs) + '\''))

    
    prof_YYYYMMDD = np.zeros((n_yyyymmdd,))
    
    for i, day in enumerate(range(n_yyyymmdd)):
        curr_day = start_date + timedelta(days=i)
        curr_day = str(curr_day) + 'T00:00:00'
        curr_day = datetime.datetime.fromisoformat(curr_day)
    
        yyyymmdd_str = curr_day.strftime('%Y%m%d')
        prof_YYYYMMDD[i] = int(yyyymmdd_str)

    prof_HHMMSS = np.tile(prof_HHMMSS, n_yyyymmdd)
    prof_YYYYMMDD = np.repeat(prof_YYYYMMDD, fac_yyyymmdd)

    nt = len(prof_HHMMSS)
    assert len(prof_HHMMSS)==len(prof_YYYYMMDD)

    time_dict=dict(zip(['prof_YYYYMMDD', 'prof_HHMMSS'], [prof_YYYYMMDD, prof_HHMMSS]))
    return time_dict, nt
    
def get_fields_dict(ns, nt, nz, fld_data_dict=None):
 
    fld_exts = ['', 'err', 'weight']

    if fld_data_dict is None:
        dummy_data = np.zeros((ns*nt, nz))
        dummy_data[:,0] = 1

        fld_strs = ['prof_T', 'prof_S']

        fld_data_keys = [fld+fld_ext for fld in fld_strs for fld_ext in fld_exts]
        fld_data_vals = [dummy_data] * len(fld_data_keys)
        fld_data_dict = dict(zip(fld_data_keys, fld_data_vals))

    return fld_data_dict, fld_strs, fld_exts

def get_prof_dict(latlon_dict, time_dict, flds_dict, tile_dict):

    ns = len(latlon_dict['prof_point'])
    nt = len(time_dict['prof_HHMMSS'])
    prof_dict = {}
    all_dicts = [latlon_dict, time_dict, flds_dict, tile_dict]
    for d in all_dicts:
        for k, v in d.items():
            if len(v) == nt:
                v = np.tile(v, ns)
            elif len(v) == ns:
                v = np.repeat(v, nt)
            prof_dict[k] = v

    prof_dict['prof_descr'] = get_prof_descr(prof_dict, nt, ns)
    return prof_dict

def get_prof_descr(prof_dict, nt, ns):
    prof_descr = []
    suffix = ['sensor000'] * nt * ns
    suffix2 = [x+'_' for x in ''.join([str(x)*nt for x in range(ns)])]
    suffix = np.char.add(suffix, suffix2)
            
    yyyymmdd_str = prof_dict['prof_YYYYMMDD'].astype(int).astype(str)
    yyyymmdd_str = np.char.add(yyyymmdd_str, np.array(['_'] * nt * ns))
    hhmmss_str = prof_dict['prof_HHMMSS'].astype(int).astype(str)
    date_array = np.char.add(yyyymmdd_str, hhmmss_str)
    
    prof_descr = np.char.add(suffix, date_array)
    return prof_descr
    
def get_tile_dict(xc, yc, prof_point, sNx=30, sNy=30):

    XC11 = np.zeros_like(xc)
    YC11 = np.zeros_like(xc)
    XCNINJ = np.zeros_like(xc)
    YCNINJ = np.zeros_like(xc)
    iTile = np.zeros_like(xc)
    jTile = np.zeros_like(xc)
    i = np.zeros_like(xc)
    j = np.zeros_like(xc)


    tile_count = 0
    for ii in range(int(xc.shape[1]/sNx)):
        for jj in range(int(xc.shape[0]/sNy)):
            tile_count += 1
            tmp_i = np.arange(sNx)+sNx*ii
            tmp_j = np.arange(sNy)+sNx*jj
            tmp_XC = xc[ np.ix_( tmp_j, tmp_i ) ]
            tmp_YC = yc[ np.ix_( tmp_j, tmp_i ) ]
            XC11[ np.ix_( tmp_j, tmp_i ) ] = tmp_XC[0,0]
            YC11[ np.ix_( tmp_j, tmp_i ) ] = tmp_YC[0,0]
            XCNINJ[ np.ix_( tmp_j, tmp_i ) ] = tmp_XC[-1,-1]
            YCNINJ[ np.ix_( tmp_j, tmp_i ) ] = tmp_YC[-1,-1]
            iTile[ np.ix_( tmp_j, tmp_i ) ] = np.ones((sNx,1)) *  np.arange(1,sNy+1) 
            jTile[ np.ix_( tmp_j, tmp_i ) ] = (np.arange(1,sNx+1) * np.ones((sNy,1))).T 
    
    tile_keys = ['XC11', 'YC11', 'XCNINJ', 'YCNINJ', 'i', 'j']
    tile_vals = [XC11, YC11, XCNINJ, YCNINJ, iTile, jTile] 
    tile_data_in = dict(zip(tile_keys, tile_vals))

    tile_dict = dict()

    for key in tile_keys:
        tile_dict['prof_interp_' + key] = tile_data_in[key].ravel()[prof_point]
    
    return tile_dict



if __name__ == "__main__":
    ungridded_lon = [167.34443606, 167.82297157, 168.68691826,
                     169.27300269, 168.48230111, 168.37412213]
    ungridded_lat = [-20.91171539, -20.64166884, -20.01145874,
                     -19.5513735,  -18.66460884, -17.80332068]
    # example below doesnt work, need to load a ds 
    # improvement: take either direct fields [xc, yc, Z] or ds with those fields
    start_date = date(1992, 1, 1)
    end_date = date(1992, 1, 31)
    freq='hourly'
    (sNx, sNy) = (6, 6)
    mitprof =  MITprof_from_metadata(grid_ds,
                      start_date, end_date, freq,
                      ungridded_lat, ungridded_lon,
                      sNx, sNy, fld_data_dict=None)

    #mitprof.tonetcdf()
