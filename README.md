# Hash Collisions with CUDA
> Modifies source data until 5 hash collisions matching an original truncated md5 hash are found.

This is my fourth time taking up the mantle on this project. 
Version 1 took over an hour to run because it wasn't random. 
Version 1.5 (rip... always remember git push) took 8 minutes once the new buffers were randomized. 
Version 2 took 40 seconds with multi-processing. 
Version 3 is still incomplete and will likeley stay that way forever since I really dont' want to bother with assembly level optimizations using Intel's math library, especially given that I don't think I'm ever going to beat version 4 once it's done.

Version 4 aims to utilize a 3060-Ti to find the collisions as quickly as possible.