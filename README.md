1. Introduction
We have introduced about find dust and more pollutant can
make harm to healthy, more in file PPT
2. Preprocess proceduce
+ Firstly, we deleted all columns unnecessary like (Url, Status,
Station name, Data Time Tz, Alert level, Station ID) and 
specified AQI index column. Simultaneously, we delete all duplicate row
because in original data we haved 3415 row but in reality we could use a few rows
+ Secondly, continue deleted all rows if them haved dash ('-'). Then we converted data time
in 'Data Time S' column to use and handle wrong 'Pressure' format
+ Thirdly, fill all mean value to blank cells of each rows, and split the coordinates
to Longitude and Latitude
+ Finally, calculated again the AQI index and draw the plot
3. Conclusion
We gained : 
+ Descriptive Statistics : We obtained the basic statistic of each pollutants, and visualized
with histogram graph
+ Trend Analysis : We draw a plot with many graph : by the time (like follow by month, day, hour),
by Longitude and Latitude (according to data, we seen many discrete data by coordinates so
we decided draw a plot with Southern and Northern) to described the trend of AQI index, and 
compared between AQI index by the time, by the coordinates.
