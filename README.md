Performing Naive Bayes on spam database implemented in Java. I split the training and test sets evenly (2301 and 2300 training instances respective) with each set having a 60-40 ratio of not-spam:spam.
I shuffled the data array so the training and test sets were getting a more even distribution of the entire database.

**Confusion Matrix (Test Set)**

|  | Predicted Spam | Actual Spam |
| ---         |      :---:      |       :---:  |
| **Actual Spam**  | 859     | 34    |
| **Actual Not Spam**     | 345       | 1062      |

Accuracy: 83.52%
Recall: 96.19%
Precision: 71.34%

The results show ~83% accuracy for the test set which is decent but clearly the model is
struggling with precision (71%) and misclassifying a good portion of the not spam as spam but
the recall (96%) is great with only misclassifying a small portion as not spam. Naive bayes
assumes all features are independent but that’s most likely not the case for this dataset, examples
of this are listed:

-  Word frequency, some of these words are likely to be used together. [! and FREE] ,
[Business, Addresses]
-  Capital letter, the three features that describe usage of capital letters are all relatively
similar. If you have a high longest_capital_run_length then you will probably also have a
relatively high capital_run_length_average.

In spite of the some of the features being non-independent, Naive bayes still does well in a few
different ways:
- Large difference between the classes. There are a large number of features where the
spam and non-spam mean difference is very large [Capital letter length]
- Easy for the model to pick up on which way to predict if it knows the feature is
clearly closer to a certain class and even more so if it is outside of 2-3 STDEVS
from the other class for a specific feature
- Large number of features. Having 57 features does help overshadow the few features that
might not be independent.

For the same reason the Naive Bayes model does good it also does poorly. The features where
there are large differences between classes are with features that are not independent [Capital
letter length]. Also, to generate a really strong spam detector you would want to look at
combinations of words, not just the frequency of a specific word. It would be a lot stronger to
have a single feature that is a combination of “FREE” + “!” instead of 2 features “FREE” and “!”
separated. The runtime of the Naive Bayes is a huge upside, the program run practically instant
on code that is not optimized for speed. 
