[//]: # (This file is automatically generated by Cute Framework's docs parser.)
[//]: # (Do not edit this file by hand!)
[//]: # (See: https://github.com/RandyGaul/cute_framework/blob/master/samples/docs_parser.cpp)
[](../header.md ':include')

# cf_haptic_create_effect

Category: [haptic](/api_reference?id=haptic)  
GitHub: [cute_haptics.h](https://github.com/RandyGaul/cute_framework/blob/master/include/cute_haptics.h)  
---

Creates a single effect instance on the device.

```cpp
CF_HapticEffect cf_haptic_create_effect(CF_Haptic haptic, CF_HapticData data);
```

Parameters | Description
--- | ---
haptic | The haptic.
data | The haptic specification.

## Remarks

Run the haptic with `haptic_run_effect`.

## Related Pages

[CF_Haptic](/haptic/cf_haptic.md)  
[cf_haptic_open](/haptic/cf_haptic_open.md)  
[CF_HapticEffect](/haptic/cf_hapticeffect.md)  
[cf_haptic_run_effect](/haptic/cf_haptic_run_effect.md)  
