# CAL supporting materials

**REFERENCE**: [MITgcm Documentation](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/cal.html)
--------------------------

In input file `data.cal`, the following settings are commonly used:
```
 $CAL_NML
 TheCalendar='gregorian',
 startDate_1=YYYMMDD,
 startDate_2=HHMMSS,
 &
```
Here, `TheCalendar` sets `usingGregorianCalendar=.TRUE.` (see `cal_set.F`). 

The `cal` package will track weekdays and leap years based on a reference date. A Gregorian year is 365 days with a day added on for leap years. The maximum number of days in a month is 31 days.
We're interested in outputting the model state and diagnostics monthly or yearly intervals (among others). Here is how to treat intervals in `data`


| interval | days | seconds |
| --- | --- | ----------- |
| 1 year | 365.25 | 31 557 600 |
| 1 month | 30.5 | 2 635 200 |
| 30 days | 30 | 2 592 000 |
| 1 week | 7 | 604 800 | 
| 1 day | 1 | 86 400 |
| 1 hour | ~0.04 | 3600 |  

In `data`, parameters like `pChkptFreq`, `chkptFreq`, `dumpFreq`, and `monitorFreq` use intervals defined in seconds. The timestep, often defined in `data` by `deltaT`, determines the number of time steps wanted in the simulation `nTimeSteps` based on the interval the model needs to run.

If using the `DIAGNOSTICS` package, you define intervals of model output by seconds in namelist `data.diagnostics`'s `frequency(n)` parameter.

--------------------------

Remember to turn on the Calendar package at runtime. In `data.pkg` add the following line: 

```
 useCal=.true.,
```

Otherwise, you will likely see an error like: 
```
CAL_GETDATE: called too early (cal_setStatus=-1 )
ABNORMAL END: S/R CAL_GETDATE
```