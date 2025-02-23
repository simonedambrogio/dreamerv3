# dreamerv3

## To do

**Main goal**
- ~~implement the environment, in a way that is as similar as possible to the original implementation~~ ✅
- ~~implement the Encoder, and test its output with the original implementation~~ ✅
    - ~~implement Space type~~ ✅
    - ~~implement RMSNorm layer in Lux~~ ✅
    - ~~implement a chain of layers using for loop~~ ✅
    - ~~check that shapes of output matches the python implementation.~~ ✅
    - ~~implements bfloat16 initialization of weights~~ ✅

- implement the Decoder, and test its output with the original implementation
    - reshape input appropriately
    - implement nn.BlockLinear in Lux


**Secondary goal**
- implementa a Flag package that transforms .yaml into a flag.parse()
