[//]: # (This file is automatically generated by Cute Framework's docs parser.)
[//]: # (Do not edit this file by hand!)
[//]: # (See: https://github.com/RandyGaul/cute_framework/blob/master/samples/docs_parser.cpp)
[](../header.md ':include')

# cf_destroy_threadpool

Category: [multithreading](/api_reference?id=multithreading)  
GitHub: [cute_multithreading.h](https://github.com/RandyGaul/cute_framework/blob/master/include/cute_multithreading.h)  
---

Destroys a [CF_Threadpool](/multithreading/cf_threadpool.md) created by [cf_make_threadpool](/multithreading/cf_make_threadpool.md).

```cpp
void cf_destroy_threadpool(CF_Threadpool* pool);
```

Parameters | Description
--- | ---
pool | The pool.

## Related Pages

[CF_TaskFn](/multithreading/cf_taskfn.md)  
[cf_make_threadpool](/multithreading/cf_make_threadpool.md)  
[cf_threadpool_kick](/multithreading/cf_threadpool_kick.md)  
[cf_threadpool_add_task](/multithreading/cf_threadpool_add_task.md)  
[cf_threadpool_kick_and_wait](/multithreading/cf_threadpool_kick_and_wait.md)  
