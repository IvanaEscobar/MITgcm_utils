# PROFILES supporting materials

**REFERENCE**: [MITgcm Documentation](https://mitgcm.readthedocs.io/en/latest/ocean_state_est/ocean_state_est.html#profiles-model-data-comparisons-at-observed-locations)
--------------------------

### Lessons learned that remain undocumented:
- When using a spherical polar grid, do **NOT** modify the `PROFILES_SIZE.h` `NUM_INTERP_POINTS` to a value under 4. Unfortunatley, `profiles_init_fixed.F` hard codes the number of interpolation points it interpolates from (see [code](https://github.com/MITgcm/MITgcm/blob/720a211d38820130cd25d86c0cf2a6770d985117/pkg/profiles/profiles_init_fixed.F#L486-L499))
- `prof_depth` should be positive valued (enforce in preprocessing?)
- Based on reading `profiles_readparms.F` there are only two fields the PROFILES packages reads, T and S


## Write your PROFILE
`pkg/profiles` uses NetCDF files to organize data. The file has a special format, which we adhere to with an xarray DataSet. This dataset has dimensions:
- `iPROF` :  Number of unique observations 
- `iDEPTH`:  Number of uqnique depth levels of each observation
  
### Generic Grid Profile


### Spherical Polar Profile
The table is a summary of what the dataset _needs_ to include for a `SphericalPolar` grid:

| field | dimension | description |
|--|--|--|
| prof_descr | iPROF | profile descriptions |
| prof_YYYYMMDD | iPROF | profile year-month-day datetime (int) |
| prof_HHMMSS | iPROF | profile hour-minute-second datetime (int) |
| prof_lon | iPROF | profile longitude |
| prof_lat | iPROF | profile latitude |
| prof_depth | iDEPTH | profile depths (+ve valued) |
| prof_[T,S] | iPROF x iDEPTH | observational data |
| prof_[T,S]weight | iPROF x iDEPTH | observational uncertainty |
| prof_[T,S]err | iPROF x iDEPTH | instrument error |

^ adapted from Goldberg, M. 2024 Zoom Meeting
