{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load NHANES data from the file\n",
    "da = pd.read_csv(\"nhanes_2015_2016.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    SEQN  ALQ101  ALQ110  ALQ130  SMQ020  RIAGENDR  RIDAGEYR  RIDRETH1  \\\n",
      "0  83732     1.0     NaN     1.0       1         1        62         3   \n",
      "1  83733     1.0     NaN     6.0       1         1        53         3   \n",
      "2  83734     1.0     NaN     NaN       1         1        78         3   \n",
      "3  83735     2.0     1.0     1.0       2         2        56         3   \n",
      "4  83736     2.0     1.0     1.0       2         2        42         4   \n",
      "\n",
      "   DMDCITZN  DMDEDUC2  ...  BPXSY2  BPXDI2  BMXWT  BMXHT  BMXBMI  BMXLEG  \\\n",
      "0       1.0       5.0  ...   124.0    64.0   94.8  184.5    27.8    43.3   \n",
      "1       2.0       3.0  ...   140.0    88.0   90.4  171.4    30.8    38.0   \n",
      "2       1.0       3.0  ...   132.0    44.0   83.4  170.1    28.8    35.6   \n",
      "3       1.0       5.0  ...   134.0    68.0  109.8  160.9    42.4    38.5   \n",
      "4       1.0       4.0  ...   114.0    54.0   55.2  164.9    20.3    37.4   \n",
      "\n",
      "   BMXARML  BMXARMC  BMXWAIST  HIQ210  \n",
      "0     43.6     35.9     101.1     2.0  \n",
      "1     40.0     33.2     107.9     NaN  \n",
      "2     37.0     31.0     116.5     2.0  \n",
      "3     37.7     38.3     110.1     2.0  \n",
      "4     36.0     27.2      80.4     2.0  \n",
      "\n",
      "[5 rows x 28 columns]\n"
     ]
    }
   ],
   "source": [
    "print(da.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Codebook for this data is:\n",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.htm#DMDEDUC2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "DMDEDUC2 is a variable which reflects a person's level of education attainment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.0    1621\n",
       "5.0    1366\n",
       "3.0    1186\n",
       "1.0     655\n",
       "2.0     643\n",
       "9.0       3\n",
       "Name: DMDEDUC2, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "da.DMDEDUC2.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5474\n",
      "(5735, 28)\n"
     ]
    }
   ],
   "source": [
    "# Finding Missing Values\n",
    "# There are Two Methods of finding Missing values\n",
    "\n",
    "print(da.DMDEDUC2.value_counts().sum())\n",
    "print(da.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We find that sum is 5474 and its shape is 5735 therefore null values will be 5735 - 5474 = 261"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can calculate missing values by pandas function isnull"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "261"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.isnull(da.DMDEDUC2).sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets replace integer codes with a text label thet reflects code's meaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Some college/AA    1621\n",
       "College            1366\n",
       "HS/GED             1186\n",
       "<9                  655\n",
       "9-11                643\n",
       "Don't know            3\n",
       "Name: DMDEDUC2x, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "da[\"DMDEDUC2x\"] = da.DMDEDUC2.replace({1: \"<9\", 2: \"9-11\", 3: \"HS/GED\", 4: \"Some college/AA\", 5: \"College\", \n",
    "                                       7: \"Refused\", 9: \"Don't know\"})\n",
    "da.DMDEDUC2x.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Relabled version of Gender variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Female    2976\n",
       "Male      2759\n",
       "Name: RIAGENDRx, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "da[\"RIAGENDRx\"] = da.RIAGENDR.replace({1: \"Male\", 2: \"Female\"})\n",
    "da.RIAGENDRx.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Some college/AA    0.296127\n",
       "College            0.249543\n",
       "HS/GED             0.216661\n",
       "<9                 0.119657\n",
       "9-11               0.117464\n",
       "Don't know         0.000548\n",
       "Name: DMDEDUC2x, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Taking the proportion of the categorical values\n",
    "\n",
    "x = da.DMDEDUC2x.value_counts()\n",
    "x / x.sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.0        0.282650\n",
       "5.0        0.238187\n",
       "3.0        0.206800\n",
       "1.0        0.114211\n",
       "2.0        0.112119\n",
       "Missing    0.045510\n",
       "9.0        0.000523\n",
       "Name: DMDEDUC2x, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Calculating the missing values\n",
    "\n",
    "da['DMDEDUC2x'] = da.DMDEDUC2.fillna(\"Missing\")\n",
    "x = da.DMDEDUC2x.value_counts()\n",
    "x / x.sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    5666.000000\n",
       "mean       81.342676\n",
       "std        21.764409\n",
       "min        32.400000\n",
       "25%        65.900000\n",
       "50%        78.200000\n",
       "75%        92.700000\n",
       "max       198.900000\n",
       "Name: BMXWT, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# generating summeries using describe \n",
    "\n",
    "da.BMXWT.dropna().describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " We can also calculate individual summary statistics from one column of data set using pandas or numpy functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "81.34267560889509\n",
      "81.34267560889509\n",
      "78.2\n",
      "78.2\n",
      "92.7\n",
      "92.7\n"
     ]
    }
   ],
   "source": [
    "x = da.BMXWT.dropna() # Extract all non-missing values of BMXWT into a variable called 'x'\n",
    "print(x.mean()) # Pandas method\n",
    "print(np.mean(x)) # Numpy function\n",
    "\n",
    "print(x.median())\n",
    "print(np.percentile(x, 50)) # 50th percentile, same as the median\n",
    "print(np.percentile(x, 75)) # 75th percentile\n",
    "print(x.quantile(0.75)) # Pandas method for quantiles, equivalent to 75th percentile"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we look at frequencies for a systolic blood pressure measurement (BPXSY1). \"BPX\" here is the NHANES prefix for blood pressure measurements, \"SY\" stands for \"systolic\" blood pressure (blood pressure at the peak of a heartbeat cycle), and \"1\" indicates that this is the first of three systolic blood presure measurements taken on a subject.\n",
    "\n",
    "A person is generally considered to have pre-hypertension when their systolic blood pressure is between 120 and 139, or their diastolic blood pressure is between 80 and 89. Considering only the systolic condition, we can calculate the proprotion of the NHANES sample who would be considered to have pre-hypertension."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.3741935483870968"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.mean((da.BPXSY1 >= 120) & (da.BPXSY2 <=139))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we calculate the propotion of NHANES subjects who are pre-hypertensive based on diastolic blood pressure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.14803836094158676"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.mean((da.BPXDI1 >= 80) & (da.BPXDI2 <= 89))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.43975588491717527\n"
     ]
    }
   ],
   "source": [
    "a = (da.BPXSY1 >= 120) & (da.BPXSY2 <= 139)\n",
    "b = (da.BPXDI1 >= 80) & (da.BPXDI2 <= 89)\n",
    "print(np.mean(a | b))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Blood pressure measurements are affected by a phenomenon called \"white coat anxiety\", in which a subject's bood pressure may be slightly elevated if they are nervous when interacting with health care providers. Typically this effect subsides if the blood pressure is measured several times in sequence. In NHANES, both systolic and diastolic blood pressure are meausred three times for each subject (e.g. BPXSY2 is the second measurement of systolic blood pressure). We can calculate the extent to which white coat anxiety is present in the NHANES data by looking a the mean difference between the first two systolic or diastolic blood pressure measurements."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.6749860309182343\n",
      "0.3490407897187558\n"
     ]
    }
   ],
   "source": [
    "print(np.mean(da.BPXSY1 - da.BPXSY2))\n",
    "print(np.mean(da.BPXDI1 - da.BPXDI2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Quantitative variables can be effectively summarized graphically. Below we see the distribution of body weight (in Kg), shown as a histogram. It is evidently right-skewed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x269bf165188>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(da.BMXWT.dropna())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next we look at the histogram of systolic blood pressure measurements. You can see that there is a tendency for the measurements to be rounded to the nearest 5 or 10 units."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x269bf8573c8>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(da.BPXSY1.dropna())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To compare several distributions, we can use side-by-side boxplots. Below we compare the distributions of the first and second systolic blood pressure measurements (BPXSY1, BPXSY2), and the first and second diastolic blood pressure measurements (BPXDI1, BPXDI2). As expected, diastolic measurements are substantially lower than systolic measurements. Above we saw that the second blood pressure reading on a subject tended on average to be slightly lower than the first measurement. This difference was less than 1 mm/Hg, so is not visible in the \"marginal\" distributions shown below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "bp = sns.boxplot(data = da.loc[:, [\"BPXSY1\", \"BPXSY2\", \"BPXDI1\", \"BPXDI2\"]])\n",
    "_ = bp.set_ylabel(\"Blood pressure in mm/Hg\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "One of the most effective ways to get more information out of a dataset is to divide it into smaller, more uniform subsets, and analyze each of these \"strata\" on its own. We can then formally or informally compare the findings in the different strata. When working with human subjects, it is very common to stratify on demographic factors such as age, sex, and race.\n",
    "\n",
    "We will consider blood pressure, which is a value that tends to increase with age. To see this trend in the NHANES data, we can partition the data into age strata, and construct side-by-side boxplots of the systolic blood pressure (SBP) distribution within each stratum. Since age is a quantitative variable, we need to create a series of \"bins\" of similar SBP values in order to stratify the data. Each box in the figure below is a summary of univariate data within a specific population stratum (here defined by age)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x269bd160dc8>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "da['agegrp'] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80]) # Create age strata based on these cut points\n",
    "plt.figure(figsize = (12, 5)) # Make the figure wider than default (12cm wide by 5cm tall)\n",
    "sns.boxplot(x = \"agegrp\" , y = \"BPXSY1\", data = da) # Make boxplot of BPXSY1 stratified by age group"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Taking this a step further, it is also the case that blood pressure tends to differ between women and men. While we could simply make two side-by-side boxplots to illustrate this contrast, it would be a bit odd to ignore age after already having established that it is strongly associated with blood pressure. Therefore, we will doubly stratify the data by gender and age.\n",
    "\n",
    "We see from the figure below that within each gender, older people tend to have higher blood pressure than younger people. However within an age band, the relationship between gender and systolic blood pressure is somewhat complex -- in younger people, men have substantially higher blood pressures than women of the same age. However for people older than 50, this relationship becomes much weaker, and among people older than 70 it appears to reverse. It is also notable that the variation of these distributions, reflected in the height of each box in the boxplot, increases with age."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x269bfca4d88>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtMAAAE9CAYAAADJUu5eAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3dfXxcdZn//9eVpE1bCgpD7dI7Aw0I2GDFWnUBLRiggALuT1fQtQGFqty0dlUUrBbYgroga1tYlRW36X4RZFUQsMa22kr5SYEApQUqNkhakvKFMkih901yff+Yk3SSTnPXOZn5zLyfj0cemXPmzJkr5zM3Vz7nOp+PuTsiIiIiItJ3JbkOQEREREQkVEqmRURERET6Scm0iIiIiEg/KZkWEREREeknJdMiIiIiIv2kZFpEREREpJ/Kch3AgTj88MO9oqIi12GIiIiISIF74oknXnP3EV3XB51MV1RUUF9fn+swRERERKTAmdmGTOtjK/Mws7FmttzM1pnZs2Y2s8v9XzMzN7PDo2Uzs/lm1mBma8zsxLhiExERERHJhjh7pluAr7r7k2Z2MPCEmS119+fMbCxwOrAxbfuzgKOjnw8AP4p+i4iIiIjkpdh6pt39ZXd/Mrr9FrAOGB3d/R/AVUD6XObnAYs8ZRXwdjM7Iq74REREREQO1IDUTJtZBfBe4FEzOxdodvenzSx9s9HAS2nLTdG6lwciRhEREZFisGfPHpqamti5c2euQ8lLQ4YMYcyYMQwaNKhX28eeTJvZcOBXwFdIlX58Czgj06YZ1vk+G5lNB6YDjBs3LnuBioiIiBSBpqYmDj74YCoqKujSsVn03J1kMklTUxNHHnlkrx4T6zjTZjaIVCJ9p7v/GhgPHAk8bWaNwBjgSTP7B1I90WPTHj4G2NR1n+5+u7tPcvdJI0bsMzqJiIiIiHRj586dJBIJJdIZmBmJRKJPvfZxjuZhwB3AOne/BcDd17r7O9y9wt0rSCXQJ7r7/wXuB6ZFo3p8ENji7irxEBEpMMlkkhkzZpBMJnMdikjRUiK9f309NnH2TJ8EfA44zcxWRz9nd7P9YuBvQAPwX8BlMcYmIiI5Ultby9q1a1m0aFGuQxGRDEpLS5k4cSITJkzg4x//OG+88QYAjY2NTJgwodO2M2fOZPTo0bS1tXVaX1dXx+TJkzn22GOZOHEin/70p9m4MTWI20UXXcSRRx7JxIkTmThxIv/4j/8IwMKFCykpKWHNmjUd+5kwYQKNjY1Aan6RqqoqqqqqOP7445k9eza7du3qiG3o0KFMnDiR448/nmnTprFnz55Yjk9XcY7m8bC7m7uf4O4To5/FXbapcPfXotvu7pe7+3h3r3J3zcYiIlJgkskkdXV1uDt1dXXqnRbJQ0OHDmX16tU888wzHHbYYdx2220Zt2tra+Pee+9l7NixPPTQQx3rn3nmGa688kpqa2v5y1/+wurVq/nsZz/bkRQD3HTTTaxevZrVq1fz5z//uWP9mDFjuOGGG/Yb2/Lly1m7di2PPfYYf/vb35g+fXrHfePHj2f16tWsXbuWpqYm7rnnngM4Cr0Xa820iIhIutra2o4erNbWVvVOi+S5D33oQzQ3N2e8b/ny5UyYMIEvf/nL3HXXXR3rv//973PNNddw3HHHdaw799xz+fCHP9zj833sYx/j2Wef5fnnn+92u+HDh/PjH/+Y++67j9dff73TfaWlpUyePLkj7ltuuYXPf/7zAKxdu5YJEyawffv2HmPpLSXTIiIyYJYtW0ZLSwsALS0tLF26NMcRicj+tLa28oc//IFzzz034/133XUXF154IZ/4xCd48MEHO8oqnn32WU48sfuJrL/+9a93lHl89rOf7VhfUlLCVVddxY033thjfIcccghHHnkk69ev77R+586dPProo0ydOhWAr3zlKzQ0NHDvvfdy8cUX85Of/IRhw4b1uP/eUjItIiIDprq6mrKy1KisZWVlnH766TmOSES62rFjBxMnTiSRSPD6669nfJ/u3r2bxYsXc/7553PIIYfwgQ98gCVLluyzXTKZZOLEiRxzzDHcfPPNHevTyzzuvPPOTo/5zGc+w6pVq3jxxRd7jNV97yjKL7zwQkfc48aN44QTTgBSCfrChQv53Oc+x0c+8hFOOumkXh+L3lAyLSIiA6ampoaSktRXT2lpKdOmTctxRCLSVXvN9IYNG9i9e3fGmum6ujq2bNlCVVUVFRUVPPzwwx2lHu9+97t58sknAUgkEqxevZrp06ezdevWXj1/WVkZX/3qV/n+97/f7XZvvfUWjY2NHHPMMcDemumGhgZWrVrF/fff37Ht+vXrGT58OJs27TPq8gFTMi0iIgMmkUgwdepUzIypU6eSSCRyHZKI7Mfb3vY25s+fz80337zPyBh33XUXP/3pT2lsbKSxsZEXX3yRJUuWsH37dq666ipuuOEG1q1b17F9X2uUL7roIpYtW8bmzZsz3r9161Yuu+wyzj//fA499NBO9x1xxBF873vf47vf/S4AW7ZsYebMmTz00EMkk0l++ctf9imWniiZFhGRAVVTU0NVVZV6pUUC8N73vpf3vOc93H333R3rtm/fzu9//3vOOeecjnUHHXQQJ598Mg888ABVVVXMmzePadOmceyxx3LSSSexbt06PvOZz3Rsn14zPXHiRHbv3t3peQcPHsyMGTN49dVXO60/9dRTmTBhApMnT2bcuHH85Cc/yRj3+eefz/bt21m5ciWzZs3isssu45hjjuGOO+7gm9/85j77PRCWXmsSmkmTJnl9vUbQExEREemtdevWdRppQ/aV6RiZ2RPuPqnrtuqZFhERkV7TDJYinSmZFhERkV7TDJYinSmZFhERkV7RDJYi+1IyLSIiIr2iGSxF9qVkWkRERHpFM1iK7EvJtIiIiPSKZrAU2ZeSaREREekVzWAp2WJmfO5zn+tYbmlpYcSIEXzsYx/r9nErVqzocZuBVpbrAERERCQM7TNYPvDAA5rBsoBc8a9f59XXXs/a/t5x+GHcestN3W5z0EEH8cwzz7Bjxw6GDh3K0qVLGT16dNZiGEhKpkVERKTXampqaGxsVK90AXn1tdd5YeRHsrfDV/7Uq83OOussfvvb3/LJT36Su+66iwsvvJCVK1cC8Nhjj/GVr3ylI9n+7//+b971rnd1evy2bdu48sorWbt2LS0tLVx77bWcd9552fs7ekllHiIiItJriUSC+fPnq1daDtgFF1zA3Xffzc6dO1mzZg0f+MAHOu479thjeeihh3jqqae4/vrrueaaa/Z5/A033MBpp53G448/zvLly/n617/Otm3bBvJPANQzLSIiIiI5cMIJJ9DY2Mhdd93F2Wef3em+LVu2UFNTw/r16zEz9uzZs8/jlyxZwv3338/NN98MwM6dO9m4ceOAT5WuZFpEREREcuLcc8/la1/7GitWrOg0CdC3v/1tTj31VO69914aGxuZMmXKPo91d371q1/tU/4x0FTmISIiIiI58fnPf57vfOc7VFVVdVq/ZcuWjgsSFy5cmPGxZ555JgsWLMDdAXjqqadijXV/lEyLiIiISE6M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\n",
      "text/plain": [
       "<Figure size 864x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "da[\"agegrp\"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])\n",
    "plt.figure(figsize=(12, 5))\n",
    "sns.boxplot(x = \"agegrp\", y = \"BPXSY1\", hue = \"RIAGENDRx\", data = da)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When stratifying on two factors (here age and gender), we can group the boxes first by age, and within age bands by gender, as above, or we can do the opposite -- group first by gender, and then within gender group by age bands. Each approach highlights a different aspect of the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x269bfc54e48>"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtMAAAE9CAYAAADJUu5eAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdfXxU5Z3//9eVWxIDJRBvwEEjAha0KAhKvxYlLQGStajU3aU/XEYRtVQoVXHVFilYdldXi1+RlV1aqAGptLVFWX8EiQVEqogBEW+gIUKUACIDKHKfTK7vH5NMc0du58zMmXk/H488yHVycuUzMZ58cp3P+VzGWouIiIiIiLReQqQDEBERERFxKyXTIiIiIiJtpGRaRERERKSNlEyLiIiIiLSRkmkRERERkTZSMi0iIiIi0kZJkQ6gPbKysmx2dnakwxARERGRGLd582aftfbc+sddnUxnZ2dTXFwc6TBEREREJMYZYz5t7LhjZR7GmB7GmLXGmO3GmI+MMVPrfXyaMcYaY7Kqx8YYM9cYU2qM2WaMGehUbCIiIiIioeDkynQl8IC1dosxpiOw2RhTZK392BjTA8gFPqt1fh7Qu/rtWmB+9b8iIiIiIlHJsZVpa+1+a+2W6ve/BrYDF1Z/+GngX4Hae5nfBCy2ARuBzsaYbk7FJyIiIiLSXmGpmTbGZAMDgHeMMaOBvdba940xtU+7ENhTa1xefWx/a75WRUUF5eXlnDp1ql0xx7oOHTrg8XhITk6OdCgiIiIiruV4Mm2MyQD+BPyUQOnHz4ERjZ3ayDHb4CRj7gbuBrjooosafEJ5eTkdO3YkOzubesm6VLPWcujQIcrLy7nkkksiHY6IiIiIaznaZ9oYk0wgkV5qrf0zcClwCfC+MaYM8ABbjDEXEFiJ7lHr0z3AvvpzWmsXWGsHWWsHnXtug+4knDp1iq5duyqRboIxhq5du2r1XkRERKSdnOzmYYCFwHZr7RwAa+0H1trzrLXZ1tpsAgn0QGvt58AKYHx1V48hwFfW2laVeNT62qF5ETFM3yN38vl8TJkyhUOHDkU6FBEREcHZlenrgH8BvmuM2Vr9lt/E+SuBXUAp8Gvgxw7GJuJKBQUFbNu2jYKCgkiHIiIiIjhYM22t3UDjddC1z8mu9b4F7nUqHrew1mKtJSFBO71LXT6fj8LCQqy1FBYW4vV66dq1a6TDEhERiWvK2Fro5ptv5uqrr+byyy9nwYIFACxcuJA+ffowbNgw7rrrLiZPngzAwYMH+cEPfsDgwYMZPHgwf/3rX4PHc3NzGThwIPfccw8XX3wxPp+PsrIy+vbty49//GMGDhzInj17yMjI4IEHHmDgwIF873vf4+DBgxF77RIdCgoKCPzNCVVVVVqdFhERiQJKplto0aJFbN68meLiYubOncvevXv55S9/ycaNGykqKmLHjh3Bc6dOncp9993Hu+++y5/+9CcmTpwIwKxZs/jud7/Lli1buOWWW/jss7/vWfO3v/2N8ePH895773HxxRdz/PhxBg4cyJYtW7jhhhuYNWtW2F+zRJeioiIqKiqAQAvI1atXRzgiERERCUuf6Vgwd+5cli9fDsCePXtYsmQJN9xwA126dAHgH//xHykpKQHg9ddf5+OPPw5+7tGjR/n666/ZsGFDcI5Ro0aRmZkZPOfiiy9myJAhwXFCQgL//M//DMBtt93GmDFjnH2BEvVyc3NZuXIlFRUVJCcnM2JEYx0mRUREJJyUTLfAunXreP3113n77bdJT09n2LBhXHbZZWzfvr3R86uqqnj77bdJS0urc7zmFn1jzjnnnCZjUPcN8Xq9FBYWAoE/trxeb4QjEhEREZV5tMBXX31FZmYm6enp7Nixg40bN3LixAneeOMNjhw5QmVlJX/605+C548YMYJ58+YFx1u3bgXgO9/5Dn/4wx8AWL16NUeOHDnr16yqquKll14C4He/+x3f+c53nHhp4iJZWVnk5eVhjCEvL08PH4qIiEQBJdMtMGrUKCorK+nfvz+PPvooQ4YM4cILL+RnP/sZ1157LcOHD6dfv3584xvfAAIlIcXFxfTv359+/frx3//93wD84he/YPXq1QwcOJDCwkK6detGx44dG/2a55xzDh999BFXX301a9asYcaMGWF7vRK9vF4v/fv316q0iIhIlDBNlR5Eu0GDBtni4uI6x7Zv307fvn3D8vWPHTtGRkYGlZWV3HLLLUyYMIFbbrnlrOefPn2axMREkpKSePvtt5k0aVJw1bq+jIwMjh075lToQHi/VyIiIiJuZozZbK0dVP+4VqbbYebMmVx11VVcccUVXHLJJdx8881Nnv/ZZ58xePBgrrzySn7yk5/w61//OkyRikSOdm0UEZFY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\n",
      "text/plain": [
       "<Figure size 864x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "da[\"agegrp\"] = pd.cut(da.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])\n",
    "plt.figure(figsize=(12, 5))\n",
    "sns.boxplot(x=\"RIAGENDRx\", y=\"BPXSY1\", hue=\"agegrp\", data=da)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Stratification can also be useful when working with categorical variables. Below we look at the frequency distribution of educational attainment (\"DMDEDUC2\") within 10-year age bands. While \"some college\" is the most common response in all age bands, up to around age 60 the second most common response is \"college\" (i.e. the person graduated from college with a four-year degree). However for people over 50, there are as many or more people with only high school or general equivalency diplomas (HS/GED) than there are college graduates.\n",
    "\n",
    "An important role of statistics is to aid researchers in identifying causes underlying observed differences. Here we have seen differences in both blood pressure and educational attainment based on age. It is plausible that aging directly causes blood pressure to increase. But in the case of educational attainment, this is actually a \"birth cohort effect\". NHANES is a cross sectional survey (all data for one wave were collected at a single point in time). People who were, say, 65 in 2015 (when these data were collected), were college-aged around 1970, while people who were in their 20's in 2015 were college-aged in around 2010 or later. Over the last few decades, it has become much more common for people to at least begin a college degree than it was in the past. Therefore, younger people as a group have higher educational attainment than older people as a group. As these young people grow older, the cross sectional relationship between age and educational attainment will change."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "agegrp    DMDEDUC2x\n",
       "(18, 30]  4.0          364\n",
       "          5.0          278\n",
       "          3.0          237\n",
       "          Missing      128\n",
       "          2.0           99\n",
       "          1.0           47\n",
       "(30, 40]  4.0          282\n",
       "          5.0          264\n",
       "          3.0          182\n",
       "          2.0          111\n",
       "          1.0           93\n",
       "(40, 50]  4.0          262\n",
       "          5.0          260\n",
       "          3.0          171\n",
       "          2.0          112\n",
       "          1.0           98\n",
       "(50, 60]  4.0          258\n",
       "          3.0          220\n",
       "          5.0          220\n",
       "          2.0          122\n",
       "          1.0          104\n",
       "(60, 70]  4.0          238\n",
       "          3.0          192\n",
       "          5.0          188\n",
       "          1.0          149\n",
       "          2.0          111\n",
       "(70, 80]  4.0          217\n",
       "          3.0          184\n",
       "          1.0          164\n",
       "          5.0          156\n",
       "          2.0           88\n",
       "          9.0            3\n",
       "Name: DMDEDUC2x, dtype: int64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "da.groupby(\"agegrp\")[\"DMDEDUC2x\"].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also stratify jointly by age and gender to explore how educational attainment varies by both of these factors simultaneously. In doing this, it is easier to interpret the results if we pivot the education levels into the columns, and normalize the counts so that they sum to 1. After doing this, the results can be interpreted as proportions or probabilities. One notable observation from this table is that for people up to age around 60, women are more likely to have graduated from college than men, but for people over aged 60, this relationship reverses."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DMDEDUC2x            1.0   2.0   3.0   4.0   5.0   9.0\n",
      "agegrp   RIAGENDRx                                    \n",
      "(18, 30] Female    0.049 0.080 0.215 0.374 0.282   NaN\n",
      "         Male      0.042 0.117 0.250 0.333 0.258   NaN\n",
      "(30, 40] Female    0.097 0.089 0.165 0.335 0.314   NaN\n",
      "         Male      0.103 0.151 0.227 0.269 0.251   NaN\n",
      "(40, 50] Female    0.106 0.110 0.173 0.313 0.299   NaN\n",
      "         Male      0.112 0.142 0.209 0.262 0.274   NaN\n",
      "(50, 60] Female    0.102 0.117 0.234 0.302 0.245   NaN\n",
      "         Male      0.123 0.148 0.242 0.256 0.231   NaN\n",
      "(60, 70] Female    0.188 0.118 0.206 0.293 0.195   NaN\n",
      "         Male      0.151 0.135 0.231 0.249 0.233   NaN\n",
      "(70, 80] Female    0.224 0.105 0.239 0.280 0.149 0.002\n",
      "         Male      0.179 0.112 0.214 0.254 0.236 0.005\n"
     ]
    }
   ],
   "source": [
    "dx = da.loc[~da.DMDEDUC2x.isin([\"Don't know\", \"Missing\"]), :]  # Eliminate rare/missing values\n",
    "dx = dx.groupby([\"agegrp\", \"RIAGENDRx\"])[\"DMDEDUC2x\"]\n",
    "dx = dx.value_counts()\n",
    "dx = dx.unstack() # Restructure the results from 'long' to 'wide'\n",
    "dx = dx.apply(lambda x: x/x.sum(), axis = 1) # Normalize within each stratum to get proportions\n",
    "print(dx.to_string(float_format = \"%.3f\"))  # Limit display to 3 decimal places"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
