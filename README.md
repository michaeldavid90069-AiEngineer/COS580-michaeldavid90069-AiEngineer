OS580-Machine-Learning/
├── Week 1 - Data Preparation/
│   ├── README.md                      <-- Discussion text, tables, and embedded plots
│   ├── Week 1DiscussionPost.ipynb      <-- Cleaned Colab notebook
│   ├── video_transcoding_sample.csv    <-- 5,000-row sample
│   └── assets/
│       ├── phik_target_correlation.png
│       ├── boxplot_before_clipping.png
│       └── boxplot_after_clipping.png


Dataset Selection & Engineering Scope

For this data prep assignment, I went with the Video Transcoding Dataset from the UCI Machine Learning Repository [I]. Working toward an AI video editing and workflow engineering role, cloud transcoding bottlenecks are a reality I think about a lot. When platforms ingest user media across random containers, you have to know how much compute power to spin up so export queues don't stall out. This dataset gives us 5,000 sampled rows, 20 features, and a target variable: utime (raw CPU transcoding time in seconds). It also gives us a solid mix of data types, pairing qualitative categories (codec, o_codec, umem) with continuous technical metrics like duration, bitrate, framerate, resolution dimensions, and frame types.

Feature Selection via Phi_K Correlation & Hypothesis Testing

Because our features mix strings with continuous numbers, I used Phi_K correlation and significance testing strictly between the input features and our utime target.

Looking at the top chart, target format requirements and motion search memory heavily dictate CPU demand. Motion estimation memory allocation umem (Phi_K ≈ 0.54), target codec o_codec (Phi_K ≈ 0.40), and target dimensions o_height / o_width (Phi_K ≈ 0.40) demonstrated the strongest predictive relationships, all with statistically significant p-values (p < 0.001). On the flip side, bidirectional b frames (Phi_K ≈ 0.00, p ≈ 0.45), input framerate (Phi_K ≈ 0.00, p ≈ 0.38), and raw file size (Phi_K ≈ 0.00, p ≈ 0.09) barely registered any signal and missed the 0.05 significance cutoff completely. Cutting these non-significant features early keeps dimensionality manageable and stops our models from learning random queue noise.

Outlier Remediation via Interquartile Range (IQR) Clipping

For outlier cleaning, I looked at the bitrate (bits per second). In real media pipelines, bitrate distributions get heavily skewed because high-bitrate master files produce massive throughput spikes compared to compressed web deliverables.

My initial boxplot showed extreme upper outliers stretching all the way to 6.0 × 10⁶ bps (or 6.0 x 10^6 bps). After running the IQR math, our upper boundary fence (Q3 + 1.5 × IQR) landed at roughly 1.43 × 10⁶ bps (or 1.43 x 10^6 bps). Instead of simply deleting those rows—which would throw away valid high-res encoding tests and shrink our data—I used Pandas .clip() to cap values at that upper threshold. This keeps all 5,000 records intact, protects our statistical sample size, and prevents extreme spikes from throwing off downstream gradient descent or clustering steps.

References

[I] T. Deneke, H. A. Haile, P. Tsigas, and M. A. A. S. O. Z. R. S. H. K. S. M. A. F. M. H., "Video Transcoding Dataset," UCI Machine Learning Repository, 2017. doi: 10.24432/C5KP53.
