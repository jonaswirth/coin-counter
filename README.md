# coin-counter
## Corner detection
Use hough lines to detect the sides of the paper. There are many matches because the lines might not be perfectly straight:
![image](doc/Hough_lines_unclustered.jpg)

Use K-Means clustering to build four clusters (one for each side of the paper) and use the cluster average.
