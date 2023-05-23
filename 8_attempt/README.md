# Attempt 8
In this attempt we changed the rewarding system. This approach rewards the states that are closer to the goal:

[[131072, 65536, 32768, 16384],

[8192, 4096, 2048, 1024],

[512, 256, 128, 64],

[32, 16, 8, 4]]

It highly rewards, if the max tile is in the top left corner and the tiles are sorted in descending order from left to right and from top to bottom.
If the max tile is not in top left corner, it still rewards the tiles that are sorted in descending order from left to right and from top to bottom but to a lesser degree.