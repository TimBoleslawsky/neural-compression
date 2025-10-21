## Data Quantity Problem in the Automobile Industry
- entry: https://arxiv.org/abs/2401.01454?utm_source=chatgpt.com,
- On-Board Automotive Systems: https://arpi.unipi.it/retrieve/5340c545-5478-40b3-ae53-38767fa5aeb7/Recent%2BAdvances%2Band%2BTrends%2B2018.pdf?utm_source=chatgpt.com
- underlying trends: https://pdfs.semanticscholar.org/3e3f/27e5c480a1dff6106e89e591c7d64ce547d0.pdf?utm_source=chatgpt.com
- vehicle computing: https://www.weisongshi.org/papers/lu22-VehicleComputing.pdf

### Useful quotes:
- "As foreseen by Intel [4], from an average of 1.5 GB of traffic data per Internet user today, we will move toward 4000 GB of data generated per day by an AD car including technical data, personal data, crowd-sourced data, and societal data". (https://arpi.unipi.it/retrieve/5340c545-5478-40b3-ae53-38767fa5aeb7/Recent%2BAdvances%2Band%2BTrends%2B2018.pdf?utm_source=chatgpt.com)
- 

## Neural/ Learned Compression 
- entry:
    - general: https://www.nowpublishers.com/article/Details/CGV-107
    - images: https://www.mdpi.com/1099-4300/26/5/357?utm_source=chatgpt.com
    - time series: https://arxiv.org/abs/2101.08784?utm_source=chatgpt.com, https://arxiv.org/abs/2412.16266?utm_source=chatgpt.com
 
## Previous Work (Simon's Thesis)
- Why is neural/learned compression relevant: “Traditional compression approaches — including domain-specific methods such as Gorilla and CHIMP — rely on hand-crafted rules and statistical assumptions about data distributions. While highly efficient for structured or stationary data, these methods lack adaptability to the heterogeneous, multimodal, and non-stationary nature of modern automotive sensor data. Learned compression methods, in contrast, can discover data-driven representations optimized for both rate and task-specific reconstruction quality.”
- Discussion of limitations of "traditional" algorithms: Dependence on manually chosen parameters like window size & Sensitivity to data characteristics (entropy, signal variability).

## Proposal Argument Chain
- Fact: There is a lot of data generated from modern vehicles!
    Assumptions about data overload leads to development of event-based observation => maintaining, ... + reduces level of observation
    Not only "autonomous" cars and vehicles.
- Why is a lot of data a problem?: Storage and transmission of data!
    -  What concretely is the data used for? What are the components that communicate through data? What needs to be accounted for? 
- Why cant we use traditional compression methods?
    - For video/image (JPEG, MP3, ...): optimized for human perception (e.g., visual quality) rather than machine learning tasks or efficient downstream data use.
    - For time series data (algorithmic approaches like CHIMP or Gorilla): Dependence on manually chosen parameters like window size & Sensitivity to data characteristics (entropy, signal variability).
- Possible Solution: Neural/ Learned Compression!
      - Current Research State: What has been done before, what has not been done before, why is it promising/ not promising?
- Our Contribution:
      - Making existing solution more efficient.
      - Looking at unexplored specific problems (losless compression).

