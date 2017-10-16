
## Approach 


Looking at the Count plot

![Count plot](https://s20.postimg.org/zfod5z819/Count.png)

**Take Aways**

* There is a hike in Footfall, which after exploring I found it was around Christmas.As it was only a few data points and we     are not predicting footfall in Decemeber, I have decided to remove these.

* The trend in data in the final stages is different from the initial trend.

**Features**

I extracted four features from the date
* **Weekday** ( the footfall is more at the weekends! )
* **Year**
* **OrdinalDay**
* **Hour**

We can notice that the footfall is low in the months of April,May,June,July,August,September,so I tried adding one more feature called "season" but it lead to some overfitting.

**Model**

I trained a **XGB** on the whole dataset which gave me a score of around 75 (RMSE) on the public leaderboard.But this model couldn't capture the trend towards the end. So I trained a basic **Linear Regression** which was fit from 12000: to capture the increasing trend.

My final submission was a weighted average of the two models,which resulted in 68.4(RMSE) on the pucblic leaderboard.
