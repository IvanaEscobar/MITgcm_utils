# TAF notes 

`genmake_local` must be a file copied directly in your `build/` diretory. Placing it in the build via a soft link, e.g. `ln -s path-to-file/genmake_local build/`, will **not** be read by `genmake2`


## Cost function accumulation
MITgcm assigns a cost function accumulation term for each type of run (1) forward, (2) adjoint, and (3) tangent linear forward.
We see these defined in `pkg/cost` header files (1) `cost.h`, (2) `adcost.h`, and (3) `g_cost.h`. 

Over time, TAF changed their naming convention, which prompted the creation of `cost_ad.flow`, which maps the common block variable names to the corresponding TAF generated variable names.
This means, for TAF generated code, `g_fc = fc_tl` and `adfc = fc_ad`. If you are tracking how the cost function changes within the TAF subroutines, search for the TAF variable names! Since these variable names a cummutative, you will never find one being set to the other, i.e. you will never see `g_fc = fc_tl` written in the source code. This is handled via the TAF flow directives.
