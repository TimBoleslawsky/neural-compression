
Thesis proposal should Include:

### What is the problem
define a research question to be answered

### what has been done before
show previous work
identify the gap we want to fill
identify relevant papers to study
show that we know the area

### how do you plan to attack the problem
what will we do
what are the challenges

### how will you evaluate the results
how do we interpret the result
what does it mean
what are the implications
what are the limitations

### why you
Courses
Background
Tim has worked on this before
We have a supervisor with a lot of experience

### what are the risks
Do we have to learn any new tools
Are we relying on existing software
What if we run into unforeseen problems
any intellectual property rights
design a minimum viable product

Proposal should not have a theory, method, result and conclusion

### Introduction
context and background
what is missing in the existing literature
research questions

### Related Work(less important)

data from vehicles can be used for downstream AI training, (why is this useful)
there is a lot of data being produced and it will likely increase (we already show this)

compressing the data is therefore essential
existing methods are fine but have drawbacks (show the drawbacks)
neural compression could be used to improve (what metrics do we want to improve)

our goal could be to increase data quality by using task aware compression algorithms (metrics for data quality, what are we comparing against: can be defined as the loss function. compete against "Learned Image Compression for Machine Perception (2021)") 

- Our goal could be to design a task aware compression algorithm
- Our goal could be to reduce the size of the compression algorithm on order to fit embedded specs 
- Our goal could be to explore model robustness 

what is the state of the art technology in the task we are performing, can we outperform it
tokenization to inform compression?

limitation: we might be removing critical data without knowing it
the downstream task may be too narrow

why is signet the customer?

- Preserving task-critical information at the expense of perceptual quality
- compressing intermediate features for split computing between edge and cloud
- creating compact, generative representations of entire datasets for efficient storage and training

This is justified by the difficulty of implementing edge solutions, which shows an interesting research gap.

Statistical methods for data compression in IoT can be useful in some contexts, particularly when the data is less structured and contain less predictable patterns.

Should we keep the network small?
should we reconstruct the data or train on the compressed data?
Should we compare against another compression algorithm which uses ML?
Model explain ability in case of critical systems?
Do we care about data reconstruction or just prediction scores?
Could we use signals from multiple sensors to predict how the data should be compressed?

The approach should not be to segment the data uniformly as we don't know how long the events will be. 

We cannot consider all possible segmentations.

Idea:
small learned predictor has a movable threshold which decided if the signal should be segmented. 
Use a learned codec to compress the data before sending it
adapt the threshold and compression
include loss function of the task and possibly the reconstruction accuracy as the distortion metric
