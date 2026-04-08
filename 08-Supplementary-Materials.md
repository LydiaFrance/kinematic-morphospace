---
number headings: auto, first-level 2, max 6, 1.1, contents ^toc
---
# Supplementary Materials ^toc

- [[#Supplementary Materials ^toc|Supplementary Materials]]
	- [[#1 Experimental Setup|1 Experimental Setup]]
	- [[#2 Marker Reconstruction|2 Marker Reconstruction]]
		- [[#2.1 Trajectories|2.1 Trajectories]]
	- [[#3 Labelling Feather Markers|3 Labelling Feather Markers]]
	- [[#4 Principal Component Analysis|4 Principal Component Analysis]]
		- [[#4.1 Scaling and Normalisation|4.1 Scaling and Normalisation]]
	- [[#5 Projection and Rotation Analysis|5 Projection and Rotation Analysis]]
		- [[#5.1 Finding Symmetrical Components|5.1 Finding Symmetrical Components]]
		- [[#5.2 Projection to Symmetrical Components|5.2 Projection to Symmetrical Components]]
		- [[#5.3 Optimal Rotation Matrix|5.3 Optimal Rotation Matrix]]
	- [[#6 Morphing Shape Modes|6 Morphing Shape Modes]]
		- [[#6.1 Separating Left and Right|6.1 Separating Left and Right]]
		- [[#6.2 Quantifying Total Variation|6.2 Quantifying Total Variation]]
		- [[#6.3 Loadings and Variance Explained|6.3 Loadings and Variance Explained]]
	- [[#7 Comparing Before and After Rotation|7 Comparing Before and After Rotation]]
	- [[#8 Missingness and Sampling Bias|8 Missingness and Sampling Bias]]
	- [[#9 Robustness to Missing Data and Labelling|9 Robustness to Missing Data and Labelling]]
		- [[#9.1 PCA eigenvalue significance: Psi and Phi statistics|9.1 PCA eigenvalue significance: Psi and Phi statistics]]
		- [[#9.2 PCA on Shuffled Dataset (control)|9.2 PCA on Shuffled Dataset (control)]]
		- [[#9.3 PCA with Marker Subsampling|9.3 PCA with Marker Subsampling]]
		- [[#9.4 PCA with Missing Labels (Pairwise Distances)|9.4 PCA with Missing Labels (Pairwise Distances)]]
		- [[#9.5 PCA with Random Relabelling|9.5 PCA with Random Relabelling]]
		- [[#9.6 Missing Data PCA (Matrix Completion)|9.6 Missing Data PCA (Matrix Completion)]]
		- [[#9.7 Alternative Decomposition Methods|9.7 Alternative Decomposition Methods]]
	- [[#10 Comparing Individual Morphing Shape Modes|10 Comparing Individual Morphing Shape Modes]]
		- [[#10.1 Individually Defined Morphing Shape Modes|10.1 Individually Defined Morphing Shape Modes]]
		- [[#10.2 Cross-validation of Shared Modes|10.2 Cross-validation of Shared Modes]]
		- [[#10.3 Bootstrap Analysis|10.3 Bootstrap Analysis]]
		- [[#10.4 Explained Variance|10.4 Explained Variance]]
	- [[#11 Visualising the Morphing Shape Modes|11 Visualising the Morphing Shape Modes]]
		- [[#11.1 Projection and Animation|11.1 Projection and Animation]]
		- [[#11.2 Morphing Shape Mode Interpretation|11.2 Morphing Shape Mode Interpretation]]
		- [[#11.3 Shape Reconstruction Accuracy|11.3 Shape Reconstruction Accuracy]]
	- [[#12 Left and Right Morphing Asymmetry|12 Left and Right Morphing Asymmetry]]
		- [[#12.1 Symmetry in Modes 1&2|12.1 Symmetry in Modes 1&2]]
		- [[#12.2 Symmetry during Flapping and Gliding|12.2 Symmetry during Flapping and Gliding]]
		- [[#12.3 Morphing Symmetry in different Flight Contexts|12.3 Morphing Symmetry in different Flight Contexts]]
	- [[#13 Time-trace Morphing Scores|13 Time-trace Morphing Scores]]
		- [[#13.1 Morphing During Varying Flight Distances|13.1 Morphing During Varying Flight Distances]]
		- [[#13.2 Obstacle versus Control Flights|13.2 Obstacle versus Control Flights]]
		- [[#13.3 Mean Scores by Individuals under Different Contexts|13.3 Mean Scores by Individuals under Different Contexts]]
	- [[#14 Evidence for  Flight Behaviour on a Continuum|14 Evidence for  Flight Behaviour on a Continuum]]
	- [[#15 Individual Comparisons|15 Individual Comparisons]]
	- [[#16 Generalising across species|16 Generalising across species]]
		- [[#16.1 Adjusting Cadaver Measurements|16.1 Adjusting Cadaver Measurements]]
		- [[#16.2 Transformation to Different Morphologies|16.2 Transformation to Different Morphologies]]
		- [[#16.3 Transformation of Morphing Shape Modes|16.3 Transformation of Morphing Shape Modes]]
	- [[#17 Bibliography|17 Bibliography]]

### List of Tables
| Label | Section | Description |
|-------|---------|-------------|
| Table A | §1 | Processed data counts by experimental period and condition |
| Table B | §1 | Processed data counts by individual hawk |
| Table C | §1 | Processed data counts by perch distance and hawk |
| Table D | §1 | Estimated wingspans and masses |
| Table E | §9.2 | Shuffle control CEV₄ results |
| Table F | §9.3 | Principal-cosines for marker-subsampled PCA |
| Table G | §10.1 | Dot-product comparison of pooled vs per-hawk modes |
| Table H | §10.1 | Min principal cosine at each subspace dimension |
| Table I | §10.1 | Tucker-congruence analysis across individuals |
| Table J | §10.4 | Cumulative explained variance by individual and condition |
| Table K | §15.1 | Cross-species cadaver wingspans |


## 1 Experimental Setup
  
We recorded a total of n = 5 captive-bred Harris' hawks flying between two perches in a purpose-built motion capture flight studio in two experimental periods for 2067 flights (Table A). The first experimental period (25 days, N=1561 flights) was previously described in KleinHeerenbrink et al 2022 (Figure A), though that work focussed solely on the whole body trajectories of flights between differently spaced perches, here we use kinematics measurements not yet described. For Period 1, four hawks flew perch-to-perch with different distances (5m, 7m, 9m, 12m) to investigate different braking strategies and capture varying flight behaviours: flapping, gliding, and landing manoeuvre. Three hawks were inexperienced males (<1 year) and one experienced female (7 years). 

![[220330_Figure1_roomschematic.png|500]]

***Figure A:** Motion capture setup for recording flights between two perches. Multiple cameras record retroreflective markers, rigid marker sets shown, and reconstruct into 3D. Reproduced from KleinHeerenbrink et al. (2022), Nature, under a CC BY license.*

| Experimental Period | Distance | Obstacle | Weight | Flights | Frames |
| :-----------------: | :------: | :------: | :----: | :-----: | :----: |
|          1          |    5     |          |        |   322   | 17575  |
|                     |    7     |          |        |   212   | 20428  |
|                     |    9     |          |        |   263   | 29647  |
|                     |    12    |          |        |   327   | 28674  |
|          2          |    9     |          |        |   189   | 22723  |
|                     |    9     |    ✓     |        |   127   | 10440  |
|                     |    9     |          |   ✓    |   67    |  7268  |
|                     |    9     |    ✓     |   ✓    |   127   |  8009  |
***Table A:** Processed data counts for the two experimental periods and conditions (perch-perch distances, with and without obstacles, and with and without worn IMU weights). Total numbers shown across hawks, 4 per experimental period. 

In the second experimental period three years later (4 days, N=510 flights): Experimental setup was replicated for the 9m flights in Period 1, but with the addition of an obstacle (N=254) to capture asymmetrical manoeuvres in the form of a turn, and/or the addition of weight onto the hawks (N=67 with weight control; N=127 with weight and obstacle). Three previously tested hawks (2 males, 1 female) and one new <1 year inexperienced male flew 9m perch-to-perch; one hawk (Rhaegal) did not participate in Period 2. By including two males who were previously recorded, we aimed to capture any individual changes that may occur with maturation and/or gain in experience. In obstacle flights, a 2m tall pole with foam padding was placed at the midpoint 4.5m from each perch; the hawks were free to fly either side of the obstacle to reach the target perch (left turns N=135, 8316 frames; right turns N=119, 10133 frames). In control flights the obstacle was removed. 
Additionally, in the second experimental period hawks wore an IMU as dead weight, similar to flight loggers commonly used in field research, to test whether the additional mass affected flight behaviour; the IMU was not used for data recording in these experiments. The hawks wore a logger board (38 g) mounted on the back, and a head-mounted IMU (2 g) which was 4.8–6.3% of body weight depending on the individual (Table D) (Shelton 2022). In control flights, no IMU was worn. All hawks in the second period experienced all experimental conditions, with/without obstacle, with/without added weight, resulting in four experimental conditions. As in the first experiments, hawks were periodically rewarded with food on arrival to the perch.


| Hawk       | Juvenile | Obstacle | IMU | Flights  |   Frames   |
| :--------- | :------: | :------: | :-: | :------: | :--------: |
| Drogon     |    ✓     |          |     |   340    |   21593    |
|            |          |          |     |    46    |    3656    |
|            |          |    ✓     |     |    32    |    1809    |
|            |          |          |  ✓  |    16    |    1014    |
|            |          |    ✓     |  ✓  |    32    |    1359    |
| Rhaegal    |    ✓     |          |     |   327    |   27900    |
| Ruby ♀     |          |          |     |   161    |   14273    |
|            |          |    ✓     |     |    32    |    2499    |
|            |          |          |  ✓  |    18    |    2197    |
|            |          |    ✓     |  ✓  |    32    |    2383    |
| Toothless  |    ✓     |          |     |   344    |   38126    |
|            |          |          |     |    50    |    9046    |
|            |          |    ✓     |     |    33    |    3948    |
|            |          |          |  ✓  |    17    |    2682    |
|            |          |    ✓     |  ✓  |    33    |    2883    |
| Charmander |    ✓     |          |     |    45    |    4453    |
|            |    ✓     |    ✓     |     |    30    |    2184    |
|            |    ✓     |          |  ✓  |    16    |    1375    |
|            |    ✓     |    ✓     |  ✓  |    30    |    1384    |
|            |          |          |     | **1634** | **144764** |

***Table B:** Processed data counts for each individual hawk, with their age, and experimental conditions. For breakdown of control flights per perch-perch distance and individual, see Supplementary Table C.*

| Distance | Hawk       | Flights |  Frames   |
| -------: | ---------- | :-----: | :-------: |
|        5 | Drogon     |   149   |   6862    |
|          | Rhaegal    |   70    |   4465    |
|          | Ruby ♀     |   41    |   2811    |
|          | Toothless  |   62    |   3437    |
|          |            | **322** | **17575** |
|        7 | Drogon     |   60    |   6012    |
|          | Rhaegal    |   72    |   7090    |
|          | Ruby ♀     |   33    |   3482    |
|          | Toothless  |   47    |   3844    |
|          |            | **212** | **20428** |
|        9 | Drogon     |   138   |   11945   |
|          | Rhaegal    |   114   |   14196   |
|          | Ruby ♀     |   81    |   8552    |
|          | Toothless  |   125   |   19117   |
|          | Charmander |   61    |   5828    |
|          |            | **452** | **52370** |
|       12 | Drogon     |   55    |   1444    |
|          | Rhaegal    |   71    |   2149    |
|          | Ruby ♀     |   24    |   1625    |
|          | Toothless  |   177   |   23456   |
|          |            | **327** | **28674** |
***Table C:** Processed data counts for each perch-perch distance by hawk for both experimental periods. Total counts in bold.*

![[2025-03-15_feathermarkers.png|300]]![[Pasted image 20250310211618.png|200]]
***Figure B:** Left: Schematic showing marker placements, adapted from KleinHeerenbrink et al. (2022), Nature, under a CC BY license. Larger retroreflective markers in a rigid asymmetrical pattern on the body allow for labelling by pattern matching. Right: Reference photograph of the left wing and tail taken after placing markers on the hawk while on the glove during feeding. Feather markers attached with non-toxic adhesive. 

In all experiments, each hawk was fitted with two rigid marker templates: one close to the centre of mass (backpack rigid markers) and one close to the base of the tail (tail-clip rigid markers) (Figure B). Each had a fixed pattern of 6.4-mm-diameter spherical retroreflective markers. The pattern of rigid markers and their attachments are described by KleinHeerenbrink et al 2022, though the rigid pattern of markers were repositioned in Period 2 to follow designs tested by (Miñano 2023) to improve tracking reliability. The hawks in some recordings wore other retroreflective markers on the head but these are not included in this analysis.

In addition, the hawks wore eight 2mm diameter, hemispherical retro-reflective markers on feathers. Each was attached with a small amount of adhesive onto the left and right longest primary feather tip (wingtip markers), the tip of the first secondary feather (secondary markers), and middle of the second primary feather (primary markers) of each wing. Markers were also placed on the outermost tail feather tips (tail-tip markers). A small amount of non-toxic paint was used to ensure consistency marker placement across sessions, and photographic references used to repeat positioning for Period 2. All markers were removed after each recording session. 

Marker positions were chosen to minimise animal handling and reduce processing complexity, while covering the hand-wing, arm-wing, and tail. We found in particular the animals would not allow any marker placement on the leading edge, opting to remove them before taking off. While these markers are insufficient for full surface reconstruction, they provide a meaningful first step in describing morphing shape changes from a large representative sample of natural flight behaviours. We opted for a low number of markers in favour of large data collection across a variety of experimental conditions, but optimising marker placement and number remains an open direction for future studies.

As only five individuals were used in total, we cannot infer general principles about how Harris hawks fly. Due to limitations in motion capture tracking of high speed flights and labelling (described below), the number of data points varies by individual and should be considered biased towards certain individuals. However we have collected an unusually large quantified kinematics dataset of natural, unconstrained flight behaviours in response to different experimental conditions, making this a unique and useful contribution to bird flight research. All data is made available.

| Name       | 2017 Wingspan (m) | 2020 Wingspan (m) | 2017 Mass (kg) | 2020 Mass (kg) |
| ---------- | ----------------- | ----------------- | -------------- | -------------- |
| Drogon     | 1.0083            | 1.0159            | 0.65           | 0.66           |
| Rhaegal    | 1.0193            | -                 | 0.62           | -              |
| Ruby ♀     | 1.0783            | 1.0613            | 0.87           | 0.84           |
| Toothless  | 1.0607            | 1.0641            | 0.74           | 0.75           |
| Charmander | -                 | 1.0164            | -              | 0.63           |
|            |                   |                   |                |                |
***Table D:** Estimated wingspans and average measured mass. Wingspans were estimated from the maximum distance between left and right wingtip markers during flight, cross-validated against physical measurements taken with wings manually extended.*
## 2 Marker Reconstruction

Marker tracking and reconstruction was achieved with a 20-camera motion capture system (Vicon Vantage 16, Vicon Motion Systems, see KleinHeerenbrink et al 2022 for further details). We used Vicon Nexus 2.7.6 software to reconstruct the positions of the markers within the flight volume, using a coordinate system corresponding to the principal axes of the flight hall. We removed any flights for which there were long sections of entirely missing backpack data (>30% of frames) or when the hawk did not land at the perch. A total of 2067 flights were included, where 1561 flights came from the first experimental period and 510 flights from the second.
Due to the challenges described by KleinHeerenbrink et al 2022 in identifying the different markers, we wrote custom software in MATLAB v2019a (Mathworks).  

After identifying the stationary markers on the perches and obstacles that were present based on a conservative tolerance, we used the spatial relationships between marker points within the same frame to determine the rigid marker patterns from the backpack and tail-pack using the Procrustes algorithm for pattern matching (see KleinHeerenbrink et al 2022 for further detail). 
### 2.1 Trajectories 

The central positions of the backpack and tail-pack were used as an estimate of the body vector, and using the destination perch position as a global origin. Starting with 939,383 frames with the central backpack position identified, we removed frames before take-off (<-0.5s, where 0s was defined by the take-off jump, see KleinHeerenbrink et al 2022) and those after landing (<0.1m horizontal distance to the perch).  We also removed 1 flight where the hawk started on the ground by filtering out flights with a starting vertical position <0m for the first 10 frames. This left 813,374 processed frames. 

 A cubic spline was fitted separately to each coordinate of the central backpack position (X, Y, and Z) per flight sequence with an RMS tolerance of 0.001 m (1 mm) to counteract the measurement error of the central position and other measurement noise. The system recorded at 120 Hz, so any interpolation gap lasting more than 30 frames (>250 ms) was removed from the time signal to avoid introducing artefacts into derivatives. This threshold was chosen to allow interpolation over brief occlusions (typical in motion capture) while excluding longer gaps where marker trajectory could not be reliably estimated. The first derivative of the smoothed trajectories was used to estimate velocity, and the magnitude of the velocity vector for total speed. The second derivative provided acceleration (Y and Z) estimates. 
 
 The whole body trajectories and kinematics for flights under each experimental condition shown here but are not used for further analysis. These include the flight paths in the XZ vertical plane (Figure C, side view) and XY horizontal plane (Figure E column 1, top down view). The velocity and acceleration were derived from the smoothing spline, and the magnitude of the velocity vector shown as total speed (Figure D), and the Y and Z components of the acceleration vector also shown (Figure E, columns 3&4).  For the side-view (horizontal versus vertical distance to the perch) position data and total speed, the binned mean for each individual hawk was also plotted alongside the raw data to demonstrate individual variation (Figures C & D, right). 
 ![[2025-02-17_FigX00_XZTraj_vector.png|500]]
 ***Figure C:** Flight trajectories from different experiments, measured at the central backpack position and splined. First four rows show trajectories from period 1: 5m, 7m, 9m, 12m perch-perch distance. Last four rows from period 2, all 9m but: control; with an obstacle; added weight; with both an obstacle and an added weight. Binned mean trajectories (bin size 0.05m) in the right column coloured by individual hawks to show individual variation.* 
  
 ![[2025-02-17_FigX01_TotalSpeed_vector.png|500]]
 ***Figure D:** Total speed of flights with different experiments, calculated using the spline at the central backpack position. Rows defined in Figure B. 


![[2025-02-22_TrajPitchAcc.png]]
 ***Figure E:** Whole body kinematics during different experiments, calculated using the spline at the central backpack position. Rows defined in Figure B. First column: trajectories using top view, the left and right turns around the obstacle are clear, though curved flights still occur in controls. Second column: body pitch estimated from the body vector which includes the rapid pitch-up manoeuvre (Carruthers 2007). Third column: estimated forward acceleration, flapping flight distinct from gliding. Fourth column: estimated upward acceleration.  

A similar spline was fitted to the central tail-pack position and used with the central backpack spline to create a estimated 3D body vector. We calculated the estimated body pitch angle using the body vector, though this estimate is noisy (order of ±5–10°) as the backpack and tail-pack were not rigidly fixed to the hawks and moved independently during flight.  The left-right component of the body vector was projected onto the floor plane and removed. The remaining vector was normalised to unit length, and pitch angle calculated via the dot product of the floor's horizontal vertical planes. For further whole body rotation estimation, see below.

For detailed analysis into trajectories with differently spaced perches in the first experimental period, see KleinHeerenbrink et al 2022. 

## 3 Labelling Feather Markers

As the spatial relationships between feather markers change with morphing, pattern matching as described in the previous section and prior work is not a suitable method for labelling feather markers. Additionally, due to marker occlusion and gaps in reconstruction, particle tracking over time is also unsuitable for labelling. Instead we used the relative distances from the feather markers to the central backpack position to label them as follows. 

For each individual, rotated relative positions of the unlabelled feather markers were overlaid for every flight (Figure F). Markers with relative distance to the central backpack beyond the total wingspan of that individual hawk were considered artefacts and removed. Each flight followed a repeatable sequence of behaviours, which could be classified based on distance to the target perch: high amplitude flapping, low amplitude flapping, gliding, landing. In the shorter flights (5m, 7m) gliding was typically absent.

![[AllUnrotatedLabelledMarkers 1.png|700]]
  ***Figure F:** Unlabelled feather marker positions relative to the central backpack position with no whole body rotation correction. Only flapping flight in straight flights shown for clarity. Columns: frontal, dorsal, side views. Body markers not shown. Measurements include morphing but also pitch, yaw and roll rotations, and individuals show left-right skew. Markers shown for each hawk: Toothless n=155,015; Drogon n=163,081; Ruby n=81,055; Rhaegal n=146,075; Charmander n=29,975. Not all markers shown here are used in analysis, only frames with all 8 markers present.

As the flights contained a significant whole body pitch rotation as part of the rapid pitch-up landing manoeuvre, the relative positions of the feather markers were rotated to remove whole body pitch for each frame. There is no accurate measure of whole body roll and yaw from the rigid markers, though these are estimated later in the analysis, see below. 
 
The labelling procedure was semi-automatic. For each individual hawk, the expected distance regions for each marker were manually defined using the anatomy of the hawk in three projection planes: xy, xz, and yz (for the local coordinate system: x points right, y points forwards, z points up). Automatic assignment then classified each unlabelled marker into these regions per frame. Since marker trajectories often overlap during the upstroke in individual planes, distances across all three planes (xy, xz, yz) were compared jointly to distinguish overlapping markers and ensure correct labelling. Duplicate markers — where two or more markers were assigned the same label within a frame — were either relabelled based on relative distance, or else removed (<5% of total frames, counted per-frame). Labelling steps were written in custom software using MATLAB v2019a (Mathworks).

There were 559,421 labelled frames from 2067 flights, with a total of 2,210,609 labelled feather markers. From these, 144,764 frames had all eight markers labelled in the same frame (26% of labelled frames). This dropout rate is expected for motion capture of fast-moving animals in a large volume with small feather markers. Incomplete frames were not used in this a. To check whether restricting to complete frames introduced systematic bias, we compared unlabelled and incomplete-frame marker distributions against the complete-frame subset and found the same geometric structure (see §8). The top of the wingbeat (wings most folded) is somewhat underrepresented in the complete-frame data but still covered. 

All analyses of the labelled markers were written in custom software using Python 3.11, which has been made open and available along with the labelled data.


## 4 Principal Component Analysis

Principal Component Analysis (PCA) identifies linear combinations that capture the maximum variance in the data. PCA is used frequently for dimensionality reduction, but its use here is to identify the dominant morphing patterns in flight and reveal coupled motions. This includes how the wings and tail move together in flight, while also distinguishing independent shape changes. This data-driven approach finds structured relationships in morphing mechanics without predefined assumptions. However, PCA assumes that linear relationships effectively describe data complexity, and so may not fully capture non-linear morphing interactions. It is critical to determine if PCA captures most of the variance, and if so its simplicity and interpretability provide advantages for analysing wing-tail morphing, informing simulations, and bioinspired design.

In this analysis, PCA is applied at several stages on different representations of the marker data: first on the bilateral (whole-shape) data to identify and correct for whole-body rotation (§5), and then on unilateral (single-side) data to extract the morphing shape modes that form the basis of all subsequent analyses (§6). In all cases, data from all individuals are pooled into a single matrix after wingspan scaling (§4.1), so the resulting components describe shared patterns across hawks rather than individual-specific ones. PCA was fitted on straight (no-obstacle) flights only; obstacle flights were then projected onto the resulting components.

We used covariance-based PCA (scikit-learn `PCA`, which computes the eigendecomposition of the sample covariance matrix via SVD). Because all variables share the same physical unit (metres, after wingspan scaling), a covariance matrix preserves the relative magnitudes of marker displacements. Using a correlation matrix instead would equalise the variance of every coordinate, inflating the contribution of markers with small displacements and obscuring the natural coupling structure. All components were retained for the initial decomposition; component retention for reconstruction and visualisation was assessed using variance-explained thresholds and cross-validated reconstruction error (§9.1 Psi/Phi tests, §10).

The covariance matrix captures how different variables co-vary with each other:

$$

C = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T

$$

• $C$: The covariance matrix
• $x_i$: The $i$-th data point (a column vector of $d$ marker coordinates)
• $\mu$: The mean of the dataset
• $n$: The total number of data points (frames)

High covariance suggests strong correlation. We obtain the eigenvalues and eigenvectors by solving:
$$
C v = \lambda v
$$
• $v$: An eigenvector (principal component direction)
• $\lambda$: The corresponding eigenvalue (variance explained by that component)

The eigenvectors represent the directions of maximum variance, which we interpret as morphing shape modes that describe shape deformations of the wings and tail in flight. The eigenvalues indicate how much variance each mode explains.

The proportion of variance explained by each principal component is given by:
$$
V = \frac{\lambda}{\sum \lambda}
$$
To transform data into the morphing shape mode space and find the scores, we use:
$$
S = P^T (x - \mu)
$$
• $P$: A matrix with eigenvectors as columns, known as the loadings matrix, with shape $[d, d]$, capturing the major modes of variation
• $x$: The original dataset of shape $[n, d]$, containing the 3D coordinates of markers per frame
• $\mu$: The mean shape across all frames, of shape $[1, d]$, subtracted to centre the data
• $S$: The transformed data in principal component space, also known as the scores

The scores matrix $S$, of shape $[n, k]$, represents the dataset in a reduced $k$-dimensional principal component space while preserving the majority of its variance.

### 4.1 Scaling and Normalisation

Marker positions are expressed relative to the central backpack position (an approximation of the centre of mass) and scaled to each hawk's maximum wingspan (Table D). This centres each frame on the bird's body and removes absolute size differences between individuals, so that PCA captures shape variation rather than positional or scale differences.

While Z-score standardisation is common practice in PCA, it is inappropriate here. All marker coordinates share the same physical unit (metres) and comparable magnitudes. Standardising each coordinate independently would equalise variances across axes, distorting the relative spatial relationships between markers and obscuring the natural couplings between wing and tail positions that are central to this analysis.

## 5 Projection and Rotation Analysis

For the bilateral rotation analysis, PCA was applied to 126,315 frames from flights without an obstacle. Each frame contained the x, y, z positions of eight feather markers plus the tailpack marker (nine markers total, $d$ = 27) relative to the central backpack position, with estimated whole body pitch removed. This resulted in 27 principal components capturing highly coupled marker positions across the entire shape of the hawk.

Since PCA was fitted on straight flights only, we projected the full dataset (144,764 frames) using the precomputed principal components and computed the cumulative sum of $V$ based on the retained components to evaluate reconstruction accuracy (Figure H).

![[Screenshot 2025-03-14 at 16.52.52.png|500]]
***Figure H:** Cumulative variance explained from bilateral PCA on data containing whole body angles, though with estimated body pitch removed. 

### 5.1 Finding Symmetrical Components

Accurately measuring whole-body roll and yaw directly was impractical with our experimental setup. Although body pitch could be estimated from the vector between the backpack and tail-pack markers, both markers exhibited slight independent movement during flight, introducing noise. Moreover, wing markers could not be used because wing motion is independent of body orientation. These factors made direct estimation of roll and yaw unreliable.

We developed the following alternative method which makes use of PCA to find highly coupled movement of markers on the hawks independent of whole body rotation. The components describe strongly correlated movements across the markers for the entire hawk shape, which include whole body rotation as well as intrinsic morphing shape changes. However, it has been previously shown that elements of morphing flight are controlled symmetrically regardless of whole body orientation, and so we expect some principal components to isolate patterns that represent intrinsic morphing only. Ros et al (2011) demonstrated in pigeons performing low-speed 90° turns that the birds steer by reorienting the whole body — acting similarly to a helicopter tilting to redirect aerodynamic forces — rather than by producing asymmetric wingstrokes. Both down- and upstrokes generated consistent forces relative to the body, indicating bilateral coupling of wing control even during turning manoeuvres. We assume this principle extends to Harris' hawks. We used a sum-of-squared-differences asymmetry score across the left-right axis of the hawk to find which principal components show morphing shape changes that exclude whole body rotation.  

First we mirrored the PCA loadings ($P$) for the left-right markers across the x axis of the body and calculated the sum of squared differences (SSD) between mirrored and original values:

$$
\text{SSD}_j = \sum_{i=1}^{d} \left( P_{ij} - \tilde{P}_{ij} \right)^2
$$

where $P_{ij}$ is the loading of the $j$-th component on the $i$-th coordinate, and $\tilde{P}_{ij}$ is the corresponding loading after mirroring the left-right markers across the body's x axis. A high asymmetry score therefore indicates whole body roll or yaw contributions to that component.

We applied a pragmatic threshold of SSD < 0.01 to classify components as symmetrical. This value was chosen empirically: an SSD of 0.01 corresponds to an RMS left-right difference of approximately 4–5 mm per coordinate, which is within the width of a single flight feather and below the noise floor of our marker placement. It is not formally derived but separates the two clearly symmetrical components (SSD = 0.0006, 0.0024) from the next component (SSD = 0.338) by over two orders of magnitude, making the classification robust to the exact threshold choice.

We found the first two components showed low asymmetry and therefore inferred they contain no whole body rotation (Figure I). As the first two components, they represent the dominant modes of morphing shape change with >83% of the total variance. The asymmetry scores for the first two components were 0.0006 and 0.0024 (RMS asymmetry of 3.9 mm and 4.8 mm respectively). In contrast, the other components showed high asymmetry scores (e.g. 0.338, 1.636; RMS asymmetry of 24.5 mm and 59.4 mm respectively)—consistent with large-scale whole-body rotation in addition to intrinsic morphing. We therefore consider the first two components to be symmetrical and without whole body rotation, unlike the rest.
![[Screenshot 2025-03-14 at 17.36.35.png|500]]

***Figure I:** Asymmetry score (sum of squared differences) for each principal component. Calculated by how different the left and right side of the hawks for each component, with low score indicating more symmetry. A threshold of score < 0.01 is shown in red, with the first two components at 0.0006 and 0.0024.

### 5.2 Projection to Symmetrical Components

We then projected the data onto these two symmetric components. The projection removes morphing detail, e.g. folding, and so the projection is not intended for morphing analysis and only for a preprocessing step to eliminate confounding rotational effects. The other components were not used.

To reconstruct the data in its original coordinate space, we applied:
$$
\hat{X} = S \times P^T + \mu
$$
where:

• $\hat{X}$ is the reconstructed dataset in 3D coordinate space.
• $S$ is the scores matrix of shape $[147532, 27]$, indicating the contribution of each principal component per frame.
• $P$ is the loadings matrix, where each column represents a principal component. Its transpose, $P^T$, maps from principal component space back to marker space.
• $\mu$ is the mean shape of the wings and tail across all frames for 9 markers in 3 dimensions $[1,27]$. It is reintroduced to restore the original coordinate frame, ensuring anatomical consistency.

### 5.3 Optimal Rotation Matrix

For each frame we therefore had the original marker positions and the symmetrical projected version. We used the Kabsch algorithm to find the optimal rotation matrix aligning the original marker positions with the symmetrical projection (Figure J). From this rotation matrix, Euler angles were extracted using the Rodrigues decomposition for interpretation.

We applied the Kabsch algorithm (Kabsch 1976), chosen as it is a form of partial Procrustes superimposition that extracts the rotation matrix but explicitly ignores translation and scaling to preserve marker-marker distances. The Kabsch algorithm solves:

$$

\min_{R} \left| R \cdot A - B \right|^2

$$

where:

• $A$ = marker positions from the symmetrical PCA projection
• $B$ = marker positions from the full reconstruction.
• $R$ = the optimal rotation matrix minimising the root-mean-square deviation (RMSD).

This rotation matrix between the symmetrical projection and original data allowed us to remove the whole-body roll, yaw, and pitch angles for each frame. We used the rotation matrix for each frame to transform the full dimensional dataset, effectively removing whole body rotations without affecting intrinsic shape changes. Using a rotation matrix avoids ambiguities of gimbal lock and has no sequence dependency. We also extracted the roll-yaw-pitch angles for each frame with the fixed rotation order X-Z-Y for interpretation only (Figure K). The X-Z-Y order was chosen because pitch (about the lateral axis) is the largest rotation in perching flight and should be applied last to minimise cross-talk between axes; roll (X) and yaw (Z) are applied first as they are smaller and more independent. Whole body pitch was added to the previous estimation of body pitch from the body vector. The symmetric projection is too low dimensional for morphing analysis and was not used further. 

![[2025-02-22_Rotation.png|500]]
***Figure J:** Finding an optimal rotation matrix using a symmetrical projection of each frame. Note the symmetric projection was used only for rotation correction and no further analysis. 

![[Screenshot 2025-03-15 at 12.49.37.png|400]]
![[Screenshot 2025-03-14 at 17.47.54.png|400]]

***Figure K:** Roll, yaw, and pitch whole body rotations, from the Rodrigues matrix of optimal rotation. Binned means shown with ±1 standard deviation shaded. Top: Experiment Period 1, whole body rotations shown for each perch-perch distance (5m, 7m, 9m, 12m). Bottom: Experiment Period 2, for obstacle and control flights. Body pitch was added to the body pitch estimation that had already been removed from the data. To see the effect of yaw on the trajectory, see Figure E first column.

We raise one limitation of this approach --  the assumption that the hawks did not independently move their wings and tail in a way that resembles roll or yaw while being unrotated. For example, this method does not distinguish roll from a theoretical morphing shape where the hawk lowers the right wing at the shoulder and raises the left, and the tail banks with the same angle through joint angle changes and not whole body rotation. Given previous findings by Ros et al 2011 we consider it more parsimonious to assume whole body rotation rather than highly coordinated, laterally anti-symmetrical control. Future work with higher numbers of markers would be required to confirm this assumption. In PCA results from markers without whole body rotation removed (see below, Figure N), we found no clear anti-symmetry in independent wing lifting and wing spreading components even in turning flights, and so consider this assumption reasonable. 

The PCA-based rotation method may introduce systematic error in cases where the projection cannot fully capture the complexity of the original data, particularly during phases of highly folded wing configurations such as the peak of the upstroke. This limitation arises because the projection onto the principal components does not capture all the variation in morphing, and underrepresents more extreme morphing configurations. However, the trade-off was deemed acceptable given the primary objective of isolating intrinsic morphing patterns while removing whole-body rotational effects. Comparing the transformed dataset distribution, it retains the overall spatial structure of the original data but with substantially reduced noise, suggesting that any introduced error is minimal. For comparison, PCA results from unrotated markers are included below (§7) and show broadly the same leading modes, confirming that the rotation correction refines rather than distorts the morphing structure.

These transformed markers were plotted in three planes of view for different individuals, where individual variation is visible in the raw data (Figure L). These results contrast with those without rotation correction which show larger distribution as they contain whole body roll and yaw (see Figure F). 
![[AllBirds_FingerprintPlots 2.png|600]]
***Figure L:** Transformed markers with estimated whole body rotations removed from each hawk. Data includes flapping, gliding, landing, turning behaviours. Total markers shown: Toothless n=462,888; Drogon n=237,720; Ruby n=172,304; Rhaegal n=226,664; Charmander n=80,680.


## 6 Morphing Shape Modes

With whole-body rotation removed, the data now contains just the intrinsic movements of the markers, capturing the relative motions of the wings and tail without the confounding effects of translation or rotation. 

### 6.1 Separating Left and Right

We separated the feather marker data into left and right sides. The left markers were mirrored in x to match the right side, so that both sides share the same coordinate frame. Midline markers (backpack and tailpack) were excluded from the unilateral analysis, as these are body-fixed reference points rather than wing or tail feather markers; they were used only for the bilateral rotation correction (§5). Each frame therefore contributes two independent rows to the data matrix — one for the right side (4 markers × 3 coordinates = 12 features) and one for the mirrored left side (12 features). This doubles the frame count from ~128k to ~257k frames for straight flights (289,528 total including obstacle flights).

Birds are configured bilaterally, and many morphing movements are symmetric or asymmetric across that axis. By making the left and right sides independent in the analysis, we avoid requiring PCA to simultaneously capture both cross-side couplings (left–right symmetry and asymmetry) and within-side couplings (e.g. wing–tail coordination), which is difficult when small timing delays or amplitude differences exist between sides. Instead, we focus PCA on the functional relationships between markers within each side and assess left–right symmetry separately by comparing scores (§11). This also avoids averaging out side-specific movements, i.e. left turns versus right turns. Rather than relying on PCA to indirectly detect asymmetry in morphing, we can directly compare left and right morphing shape mode scores to quantify how they are used together, independently, or in opposition. PCA results from whole shape are not used further.

### 6.2 Quantifying Total Variation

The raw wing-tail shapes are widely distributed around the mean (mean Euclidean distance = 0.241 ± 0.128 m, normalised to maximum wingspan). Log-determinant analysis quantifies how much of the total geometric variability in all 3D marker coordinates are captured by the first few PCA axes. Each eigenvalue of the covariance matrix (score) represents variance along one independent direction (mode), and so taking the product of all eigenvalues and its logarithm measures the total "volume" that the data occupy in the space. When the same data is computed after projection into the first four modes, the log-determinant reduces from −84.82 to −19.59 (natural logarithm; units are log m² since all coordinates are in metres after wingspan scaling). This reduction of ~65 log units corresponds to a covariance-volume ratio of e⁶⁵ ≈ 10²⁸, confirming that almost all variation is confined to a small number of axes. In other words, while the raw coordinates vary widely,  almost all variation lies amongst a small number of axes directions (modes). 

A permutation test (2,000 iterations) in which the coordinate columns were independently shuffled produced no instance of such a large drop (permutation p < 0.001), confirming that this concentration of variance is not an artefact of data scale or sampling but a genuine geometric property of the hawks' shape changes. Therefore we find a quantitative measurement that the complexity of their flight collapses to a low-dimensional manifold.

### 6.3 Loadings and Variance Explained

We ran PCA on corrected feather marker data from flights without an obstacle from both experimental periods from all hawks (252,630 frames). As in the PCA for whole body shape, our input data set $x$ is now shape $[257116,12]$ with just straight flights, with  $\mu$ $[1,12]$ to represent one side of the hawk. $P$, with the eigenvectors as columns, known as the loadings matrix, has shape $[12, 12]$, capturing the major modes of variation. We excluded the obstacle flights from defining the principal components due to the lower number of data points (37,948 frames) and prevent unequal weighting in the definitions. We then projected the entire dataset (289,528 frames) onto these components, and due to the high explained variance we confirm these morphing shape modes are suitable for describing the morphing in the obstacle flights (Figure M).

We refer to the eigenvectors or components from this analysis as "morphing shape modes", as they contain the dominant patterns of morphing from the hawks' flights (Figure M). 

![[Screenshot 2025-03-14 at 17.51.00.png|500]]

![[Screenshot 2025-03-14 at 17.50.51.png|400]]
***Figure M:** Top: Cumulative variance explained for the morphing shape modes, as performed on unilateral morphing shapes. Bottom: Heatmap of absolute principal component (PC) loadings for each feather marker. Each row represents a marker and its dimension, and each column represents a morphing shape mode. Darker colours indicate higher absolute loading values, signifying stronger contributions of a marker. Annotations indicate cumulative variance explained thresholds.

## 7 Comparing Before and After Rotation

To assess the effect of the rotation correction (§5), we ran PCA on unilateral marker data both with and without the Kabsch rotation correction and compared the resulting components.

The cumulative explained variance profiles are similar in both cases: the first four modes capture approximately 96% of variance regardless of rotation correction. The correction slightly sharpens the variance concentration because whole-body rotation is removed from the residual.

The absolute dot products between corresponding modes (Figure N, diagonal) show that the first two modes are virtually identical (dot products ≥ 0.999), confirming that wing lifting and wing spreading — the dominant morphing patterns — are unaffected by the rotation correction. Modes 3–4 remain well aligned (0.983–0.992). Higher modes show progressively lower alignment (dot products 0.592–0.903), consistent with the removal of whole-body rotation from modes that previously mixed rotation with morphing. This decay is expected: lower-variance modes explain less of the total variation and are therefore more susceptible to contamination by whole-body rotation; the correction disentangles these contributions rather than altering the underlying morphing structure.

Principal-cosine analysis (sweeping subspace dimension *k* = 1…6, where no values are trivially guaranteed; see §9 for method) confirms genuine alignment of the leading subspaces. The rotation correction does not create new structure; it separates existing morphing variation from whole-body rotation, resulting in cleaner modes from mode 3 onwards. This is why the corrected data is used for all subsequent analysis.

![[Screenshot 2025-03-14 at 19.11.52.png|250]]
***Figure N:** The absolute values of dot products between principal components before and after correction, indicating the degree of alignment between the respective bases. Dot product diagonal values: 0.999, 0.999, 0.992, 0.983, 0.903, 0.900, 0.852, 0.797, 0.758, 0.706, 0.651, 0.592.*


## 8 Missingness and Sampling Bias

*Notebook reference: `07_MissingnessAndSamplingBias.ipynb`*

### Overview

The primary PCA analysis reported in the main text is based upon approximately 126,000 bilateral frames where all eight markers were successfully reconstructed and labelled. However, the motion-capture system recorded a substantially larger dataset of approximately 559,000 bilateral frames, many of which contain partial observations with one or more markers missing. This section examines whether the complete-marker subset exhibits systematic sampling bias, and whether such bias, if present, compromises the robustness and generalisability of the morphospace model.

Marker visibility is not random: gliding configurations, characterised by outstretched wings and tail in relatively stable poses, are systematically easier for the camera system to detect on feather markers. Conversely, configurations involving rapid wingbeats and extreme wing folding tend to yield partial observations, as distal markers become occluded by the bird's body or move outside the capture volume. This raises a fundamental concern: does the complete dataset oversample gliding-like configurations and thus bias the derived morphological modes?

### Flight-Phase Specificity of Variance Explained

To assess whether sampling bias inflates or deflates the principal components, we stratified the marker trajectories by flight phase and computed independent PCA decompositions for each phase. Flight phases were defined by behavioural annotation: phase 1 (takeoff, large flapping), phase 2 (cruise, smaller flapping), phase 3 (transition), and phase 4 (glide and pitch-up landing).

Cumulative explained variance (CEV) in the first four principal components was highest in the flapping phases: phase 1 and 2 each achieved approximately 97% CEV₄. The glide-and-landing phase (phase 4) showed the lowest CEV₄ at approximately 91%. The combined dataset across all phases achieved 96.3% CEV₄. Critically, this combined value is *depressed* by phase 4, not inflated, indicating that the gliding bias concern is unfounded.

When projecting phase-specific data into the shared PCA basis (computed from the complete dataset), principal components 1 and 2 (corresponding to wing lifting and wing folding) showed eightfold reduction in amplitude from takeoff to gliding. By contrast, principal components 3 and 4 (wing sweep and tail spread) remained approximately constant across all phases. The lower CEV₄ observed in phase 4 reflects the attenuation of the flapping signal rather than an artefact of gliding oversampling. These findings directly refute the hypothesis that gliding bias artificially inflates the PCA results, and instead suggest that the 96.3% CEV₄ reported in the main text represents a *conservative* estimate of morphospace dimensionality.

### Spatial Distribution: Complete versus Partial Frames

To quantify the distribution mismatch, we compared the spatial occupancy of complete frames (all four unilateral markers present) against partial frames (at least one marker missing, but the marker in question present) across two orthogonal projections: the x–z plane (wing span and vertical excursion) and the x–y plane (wing span and lateral deflection).

For each of the four unilateral markers, we computed the bin occupancy density in both complete and partial subsets. We then identified the densest quartile of bins—corresponding to the gliding morphology (spread wings, elevated tail)—and counted the proportion of frames falling within this dense region.

Results showed that 80% of complete frames concentrated within the densest quartile, whereas only 67–79% of partial frames did so—a systematic shift of 8–13 percentage points. This difference was consistent across all four markers. Conversely, partial frames populated configurations corresponding to deeper wingbeats and extreme wing folding more uniformly than complete frames. Two-dimensional density maps (Figure S8a, b) reveal that red regions (complete-frame excess) cluster around the gliding peak, whilst blue regions (partial-frame excess) appear at more extreme wing-folding angles. This pattern is consistent with the hypothesis that distal feather markers tuck behind the body during rapid wing motion, systematically generating partial observations in folded-wing configurations.

![Spatial density of marker positions: x–z plane (wing span and vertical displacement) comparing complete frames (all markers visible) against naturally partial frames (one marker missing). Red indicates higher density in complete frames; blue indicates higher density in partial frames.](figures/supplementary/S07_density_xz.pdf){width=90%}

![Spatial density of marker positions: x–y plane (wing span and lateral deflection) comparing complete frames against naturally partial frames.](figures/supplementary/S07_density_xy.pdf){width=90%}

Notably, the difference maps revealed mislabelling artefacts in the wingtip row: isolated red peaks in regions where wingtip markers should not occur, accompanied by structural inconsistencies, indicate that some labelled wingtip markers are actually primary-feather markers. Nevertheless, the overall distribution structure—the relative density gradient, the location of the gliding peak, and the clustering of folded-wing configurations—is robust and consistent between complete and partial frames. This structural consistency suggests that sampling bias affects *density*, not the fundamental geometry of morphospace.

### Occlusion Robustness

A critical test of morphospace robustness is whether the principal axes remain invariant when markers are systematically removed. We compared two PCA bases: (A) computed from the complete dataset with one marker artificially removed via masking, and (B) computed from naturally occurring frames where that same marker was genuinely missing.

For modes 1 and 2 (wing lifting and wing folding), the direction cosines between complete-frame PCA and occluded-frame PCA exceeded 0.99 in all cases, indicating that the dominant morphing modes are virtually identical regardless of whether occlusion is synthetic or natural. Mode 3 (wing sweep) exhibited greater sensitivity, with cosines ranging from 0.73 to 0.96, particularly when the wingtip was occluded. However, since modes 1 and 2 jointly account for more than 90% of total variance, the subspace spanned by these two modes is effectively invariant to occlusion patterns observed in the empirical data.

To evaluate projection reliability, we computed PC1 and PC2 scores for partial frames using the complete-frame basis and correlated these projections against scores recomputed from the full marker set (where available). With one marker masked, Pearson correlations exceeded 0.99 for both PC1 and PC2. With two markers masked simultaneously (the worst-case scenario), the minimum correlation fell to 0.96, with root-mean-square error (RMSE) at approximately 23% of the within-phase standard deviation of the score. Projection introduces noise but not systematic bias.

### Coverage Analysis

We assessed the extent to which naturally partial frames explore regions of morphospace not represented in the complete dataset. Among all bins occupied by partial frames, fewer than 1% fell in novel bins not present in the complete-frame occupancy map. This finding indicates that the sampling bias is fundamentally *density-based*: partial frames do not unlock new morphological regions, but rather provide denser coverage of folded-wing and intermediate configurations already present (albeit sparsely) in the complete dataset.

### Marker Dropout Patterns and Detection of Mislabelling

Marker dropout follows a consistent anatomical and postural logic. When the wingtip marker is missing, the remaining wing markers indicate a more folded configuration. When the tailtip marker is missing, the remaining wing markers show more extreme tail deflections. This co-dropout structure reflects shared biomechanical causes (e.g., rapid wingbeat driving distal-marker occlusion) rather than random sensor failures.

We exploited the anatomical ordering constraint to detect systematic mislabelling: the ordering along the wing should invariably be secondary (proximal) < primary < wingtip (distal) in the lateral (x) direction. Violations of this ordering flag potential label swaps. We mapped violations as a function of position in morphospace.

![Density of marker ordering violations (secondary < primary < wingtip along x axis) in two projections. Violations cluster in specific regions corresponding to rapid-movement configurations, particularly deep wingbeats and extreme lateral deflection.](figures/supplementary/S07_ordering_violations.pdf){width=90%}

Ordering violations clustered in regions of coordinate space corresponding to the density mismatches identified above, corroborating the finding that partial frames more uniformly explore folded-wing and extreme-motion configurations. Notably, violation rates were slightly elevated in complete frames, suggesting that when fewer markers are visible, the reconstruction software has fewer candidate matches and thus makes fewer labelling errors.

### Summary

Systematic assessment via multiple lines of evidence confirms that marker dropout is not random, and the complete-marker dataset does systematically oversample gliding-like postures (8–13 percentage points higher density in the gliding peak). However, this sampling bias does not compromise morphospace structure: principal modes remain invariant whether computed from complete frames or from naturally occluded data; projections remain stable under marker removal; and morphospace coverage is identical between complete and partial subsets. The lower CEV₄ in gliding phases (91%, versus 97% in flapping) directly refutes the concern that gliding bias inflates results; instead, it demonstrates that the 96.3% CEV₄ reported in the main text is conservative.

These findings establish that the morphospace model is robust to the sampling biases inherent in optical motion-capture data, and that conclusions drawn from the complete-marker subset are generalisable to the full population of flight configurations.


## 9 Robustness to Missing Data and Labelling

We performed a suite of controls to verify that the identified morphing-shape modes reflect genuine coordinated shape change rather than artefacts of marker layout, labelling, missing data or PCA ordering. Tests included shuffle controls, marker subsampling, label-free distance representations, simulated labelling errors, matrix-completion for extensive dropouts, per-individual PCA comparisons and stratified bootstraps. Across these methods the pooled PCA basis was stable: the first four modes consistently explained the vast majority of variance (≈0.965 cumulative explained variance in the labelled dataset), reconstructed held-out frames with small RMS errors (<0.015 m), and aligned closely to per-hawk bases (principal-cosines ≈1.0 for modes 1-4). Full methods, figures and numeric details follow below.


### 8.1 PCA eigenvalue significance: Psi and Phi statistics

To assess the suitability and robustness of the principal components and verify the results reflect meaningful structure rather than spurious variance, we conducted randomisation tests following the recommendations of Bjorklund (2019). Since PCA always produces ordered components, even from random data, these tests ensure the extracted shape modes represent true underlying morphing patterns rather than statistical artefacts. 

As described by Björklund et al, the Psi statistic quantifies how much the observed eigenvalues deviate from a uniform distribution, indicating whether the principal components differ significantly from those expected by chance. A high Psi value suggests that the variance captured by the principal components is meaningfully structured rather than randomly distributed. In this analysis, the Psi value was 11.85 with a p-value of 0.0005, strongly indicating that the identified principal components are statistically distinct.

Since PCA was performed without Z-score normalisation, the relative variance structure was preserved, retaining the spatial relationships between markers rather than artificial equalising of variance across dimensions. This aligns with our goal of capturing coordinated morphing shape changes rather than normalising independent marker variability. However, since eigenvalues are sensitive to scale, this may affect how the standard randomisation tests interpret significance.

The Phi statistic provides a measure of the spread of eigenvalues and is typically used when the data are normalised prior to PCA, where eigenvalues are expected to be approximately 1.0 under the null hypothesis. The Phi value was 0.4471, with a high p-value approaching 1.0. Since our data are not Z-score normalised, eigenvalues are not centred around 1.0 and the standard Phi statistic is not appropriate. We therefore also computed an alternative Phi statistic (Φ_alt) that first normalises eigenvalues by their mean before applying the Phi formula:

$$
\Phi_{\text{alt}} = \sqrt{\sum_{i=1}^{p} \left( \frac{\lambda_i}{\bar{\lambda}} - 1 \right)^2}
$$

where $\bar{\lambda} = \frac{1}{p}\sum \lambda_i$. This makes the statistic scale-invariant and suitable for non-standardised data. The Φ_alt value was 1.4924 (p = 0.0005), confirming that the concentration of variance in the leading components is statistically significant and not an artefact of data scale.

While the high standard Phi p-value could suggest later components are dominated by noise, qualitative inspection of the mode shapes reveals that some of these smaller modes capture fine-scale handwing and tail adjustments, aligning with previously described independent observations of bird flight kinematics (Gillies et al 2011, Harvey et al 2022a). We therefore retained modes on a qualitative basis — whether the projected shape change was biomechanically interpretable — rather than applying a fixed variance threshold. This suggests that even low-variance modes likely play a role in morphing flight control despite their minimal contribution to the total variance.

To ensure the extracted modes were meaningful, we visually examined their structure in comparison to established flight mechanics. Modes with low variance explained and resembled noise on projection should be considered with care, namely modes 10-12. With this caveat, we consider the PCA results robust for this analysis. In future work, alternative dimensionality reduction methods could be explored to better capture subtle morphing control dynamics beyond PCA’s linear constraints. ICA (a linear method that seeks non-Gaussian sources rather than maximum-variance directions) and manifold learning approaches (e.g. UMAP, diffusion maps) are candidates, though as shown in §9.7, ICA’s objective is poorly matched to this data.

### 8.2 PCA on Shuffled Dataset (control)

To test whether the observed eigenvalue spectrum arose from coordinated marker motion rather than layout or sampling, we applied three shuffle controls (2,000 repeats each), each destroying a different level of structure:

1. **Coordinate shuffle**: each of the 12 feature columns (4 markers × 3 axes) was permuted independently across frames, destroying all inter-marker and inter-axis correlations while preserving each column's marginal distribution.
2. **Vector shuffle**: each marker's 3D position vector (x, y, z together) was permuted independently across frames, preserving within-marker spatial structure but destroying inter-marker coordination. Any variance concentration beyond this null indicates genuine inter-marker coupling.
3. **Label shuffle**: within each frame, the four marker identities were randomly reassigned, preserving the instantaneous frame geometry but breaking consistent anatomical identity across time.

In the real data the first four modes explain 96.5% of variance. All three null distributions fall far below, with none of the 2,000 permutations ever reaching the observed value (permutation *p* < 0.001 for all):

| Shuffle type | Null CEV₄ (mean) | Observed CEV₄ |
|-------------|:-:|:-:|
| Coordinate | 0.771 | 0.965 |
| Vector | 0.816 | 0.965 |
| Label | 0.753 | 0.965 |

***Table E:** Cumulative explained variance (CEV₄) for the first four modes under three shuffle controls (2,000 permutations each, all p < 0.001). The observed data far exceed all null distributions.*

The hierarchy of null distributions is itself informative. The label shuffle produces the lowest null CEV because it preserves within-frame geometry but destroys the temporal consistency of marker identity, confirming that the steep observed spectrum depends on markers tracking the same anatomical points over time. The vector shuffle sits slightly higher, reflecting modest within-marker 3D correlations, e.g. a wingtip moving predominantly along one axis. The large gap between even the vector shuffle null and the observed data confirms that inter-marker coordination (and not just individual marker variance) drives the low-dimensional structure.

![[S08_shuffle_controls.pdf]]

***Figure G:** Shuffle controls. Observed cumulative explained variance (blue) compared against three null distributions (grey shading = 95% CI, dashed = null mean). Each null model destroys a different aspect of marker coordination. The observed data lie far above all null distributions, confirming the morphing structure reflects genuine wing-tail coordination (2,000 permutations each; all p < 0.001).*


### 8.3 PCA with Marker Subsampling

To assess the robustness of the principal components to changes in marker configuration, we reran PCA on all combinations of 3 out of 4 available markers. Although dropping one marker reduces the available shape information by 25%, the resulting PCA remained nearly identical to the full analysis. For each combination of three markers (leave-one-out across 4 markers), PCA was repeated to estimate the stability of the variance explained. The cumulative variance curves for every triplet overlapped closely and rose steeply, showing that dimensionality reduction is preserved regardless of which marker is missing. 

![[Screenshot 2025-11-04 at 16.43.08.png|500]]

Principal-cosine analysis confirmed that the dominant modes are unaffected by marker loss.  For all four triplets, Modes 1-3 were identical (cos θ = 1.0), and the fourth mode showed high but slightly variable alignment depending on which marker was omitted (cos θ ≈ 0.65–1.0).  These results demonstrate that the principal patterns of shape change are stable and reproducible even when one marker is missing, indicating that the morphing geometry does not depend on any single marker configuration.

| Included Markers            | Mode 1 | Mode 2 | Mode 3 | Mode 4 |
| --------------------------- | :----: | :----: | :----: | :----: |
| primary, secondary, tailtip | 1.000  | 1.000  | 1.000  | 0.654  |
| wingtip, secondary, tailtip | 1.000  | 1.000  | 1.000  | 0.999  |
| wingtip, primary, tailtip   | 1.000  | 1.000  | 1.000  | 0.999  |
| wingtip, primary, secondary | 1.000  | 1.000  | 1.000  | 0.744  |
***Table F:** Principal-cosines between the full and marker-subsampled PCAs.  Values of 1.0 indicate identical axes.  The first three modes are invariant to marker removal, while the fourth mode varies slightly depending on which marker is omitted (mean = 0.849 ± 0.16).*


In all four marker triplets, the first four components together explained more than 95% of the total variance, closely matching the full four-marker result.  This confirms that the low-dimensional structure of morphing flight is not an artefact of the marker layout.

### 8.4 PCA with Missing Labels (Pairwise Distances)

In this control we represent the data instead as six pairwise distances among the four 3D markers on each side of the hawks, binned into forty equal-width histogram bins (bin width chosen to resolve the range of observed pairwise distances at a resolution finer than marker placement error). As this representation is invariant to marker identity, translation, and rotation, it provides a label-free description of shape. PCA on these histograms produced a similar steep cumulative variance profile as the standard labelled PCA ($CEV_4 = 0.97$ vs $0.96$), confirming the low dimensional structure arises from the geometry of the markers rather than labelling convention. 

The labelled frames come from a larger, unlabelled dataset of all detection by the motion capture that were not rigid-body marker sets, alongside spurious "ghost" markers ($\approx7$ million detected points). We tested whether the process of labelling found a subset with substantially different geometry, i.e. that the labelling process imposed artificial structure. We found the same distribution of pairwise distances (Wasserstein distance = 0.0057 between mean-normalised histograms; the distance is in the same normalised units as the histogram bins, so a value of 0.0057 indicates that the two distributions differ by less than 1% of the typical inter-marker distance), with the same low dimensional variation collapse ($CEV_4 \approx 0.93$ unlabelled vs $0.96$ labelled). When projected into the shape space defined by all detections, the fully labelled frames occupy the same manifold.  As the unlabelled dataset contains the labelled frames, this overlap confirms that marker labelling did not create new structure but simply identified the highest-quality region of the same geometric space.

### 8.5 PCA with Random Relabelling

Marker labelling in motion capture is not error-free: automated label assignment can occasionally swap marker identities between frames (see §3). If the morphing structure were fragile — dependent on perfect frame-by-frame labelling — even a small fraction of swapped labels would collapse the PCA basis. We simulated labelling errors by randomly permuting marker identities in a controlled fraction of frames, then recomputed PCA and compared with the baseline via principal cosines (200 repeats per error rate).

At 5% relabelling, modes 1–2 are virtually unchanged (principal cosines > 0.999), and mode 3 remains well aligned (cosine ≈ 0.87). $CEV_4$ drops from 0.965 to 0.910, reflecting noise spreading into additional components rather than a structural change in the dominant morphing subspace. The overall PCA structure is preserved.

At 25% relabelling — far beyond any realistic error rate — modes 1–2 still show cosines > 0.99, demonstrating that the dominant wing sweep and fold axes are robust to labelling noise. Mode 4 degrades substantially (cosine ≈ 0.23), consistent with its lower eigenvalue making it more susceptible to perturbation.

The subspace degrades gracefully and predictably: higher-variance modes are more robust, as expected from perturbation theory of symmetric matrices. These results confirm that the leading morphing modes are not artefacts of precise labelling.

### 8.6 Missing Data PCA (Matrix Completion)

A larger dataset from the hawks of 936,806 unilateral frames contained marker dropouts (each marker had between 21-33% dropout). As conventional PCA requires complete data, we used a matrix-completion approach that estimates missing values by exploiting the low-rank structure of the data. We applied scikit-learn's `IterativeImputer` (`max_iter=100`, `tol=1e-3`, default BayesianRidge estimator), a multivariate imputation by chained equations (MICE) method. MICE iteratively models each feature as a function of all other features, imputing missing values one feature at a time and cycling until convergence. Unlike expectation-maximisation PCA (which assumes a single low-rank model), MICE fits a separate regression for each feature column, making it more flexible but not constrained to a PCA-like subspace. To assess reconstruction accuracy, we randomly masked 10% of the observed entries, re-imputed them, and compared the recovered values to the true coordinates. The cross-validation RMS error was 0.043 m (4% of the wingspan) indicating the imputer recovered missing points with high fidelity.

PCA performed on this artificially completed dataset shows the same steep decline in explained variance as the fully labelled dataset ($CEV_4 \approx 0.96$ vs $0.96$). The additional noise from imputation had minimal effect on the cumulative variance; the shape of the curve and relative contributions of the first four components were preserved. Principal-cosine analysis between the components yielded $\cos \theta_1 = 0.9998$ and $\cos \theta_2 = 0.9991$, while lower variance modes were also well aligned ($\cos \theta_3 = 0.9942$ and $\cos \theta_4 = 0.9525$). The first three modes are near-identical, while the fourth mode shows moderate alignment, reflecting the greater sensitivity of lower-variance modes to imputation noise. These results confirm that the dominant axes remain stable even with large amounts of missing data, and so the hawks' wing-tail geometry remains inherently low dimensional.

We do not use the artificially completed dataset for any further analysis here.

### 8.7 Alternative Decomposition Methods

To confirm that the low-dimensional structure is a property of the data rather than an artefact of PCA, we compared four alternative decomposition methods against standard PCA: z-score normalised PCA, Independent Component Analysis (ICA), Sparse PCA, and Factor Analysis (FA). Each method was fitted on the same centred straight-flight training data (252,630 frames, 12 features) and evaluated on held-out turning flights. Subspace alignment was quantified using principal cosines at each subspace dimension *k* = 1…6 (restricting to *k* ≤ *n*/2 = 6 avoids trivially guaranteed alignments). Reconstruction quality was measured as RMSE on the held-out turning data.

The methods split into two groups. **Factor Analysis** (min cosine at *k* = 4: 0.96) and **Sparse PCA** (α = 0.1; min cosine at *k* = 4: 0.89) recovered subspaces closely aligned with standard PCA, confirming that the dominant morphing structure is robust. FA's per-feature noise was negligible (mean Ψ = 0.0002), and Sparse PCA preserved the leading subspace even with 66% of loadings zeroed. **Z-score PCA** (min cosine at *k* = 4: 0.27) and **ICA** (min cosine at *k* = 4: 0.85) found substantially different subspaces, but for understood reasons that favour standard PCA:

- **Z-score PCA** equalises feature variances before decomposition. This is appropriate when features have different physical units, but here all twelve features are marker coordinates in the same unit (metres, normalised to wingspan). Equalising variances discards the meaningful variance hierarchy, inflating the influence of near-stationary markers and diluting that of highly mobile ones (see also §4.1). The divergence therefore confirms that variance differences between markers are physically informative, not artefactual.
- **ICA** seeks maximally non-Gaussian sources. The leading PCA modes (1–4, ~96% of variance) are near-Gaussian (|kurtosis| < 0.5), so ICA has limited non-Gaussian structure to exploit in the dominant subspace. It achieves moderate alignment (cosine = 0.85) but with substantially higher reconstruction error (RMSE = 0.068 m vs 0.014 m for PCA), consistent with its objective targeting statistical independence rather than variance.

FA also illustrates why PCA is preferable for interpretation: FA loadings are identifiable only up to an arbitrary orthogonal rotation (rotational indeterminacy), factors lack a natural variance-based ordering, and partial reconstruction (using a subset of factors) is not exact as it is with PCA's orthogonal truncation.

| Method | Min cos (*k* = 4) | RMSE (*k* = 4) | Notes |
|--------|:---:|:---:|-------|
| Standard PCA | 1.00 | 0.0143 | Reference |
| Factor Analysis | 0.96 | 0.0147 | Negligible noise (Ψ ~ 0.0002) |
| Sparse PCA (α = 0.1) | 0.89 | 0.0293 | 66% near-zero loadings |
| Z-score PCA | 0.27 | 0.0207 | Homogeneous units make z-scoring inappropriate |
| ICA | 0.85 | 0.0681 | Leading modes near-Gaussian; higher reconstruction error |

Standard PCA achieves the lowest out-of-sample reconstruction error and provides a unique, variance-ordered, orthogonal decomposition. The divergent methods do not reveal hidden structure that PCA misses; rather, they confirm that PCA's assumptions — features in common units with a meaningful variance hierarchy and near-Gaussian leading modes — are well satisfied by the morphing data.

![[S8_alternative_methods_comparison.pdf]]
***Figure G:** Comparison of alternative decomposition methods against standard PCA. Top row: minimum principal cosine at each subspace dimension k (1.0 = identical subspace). Bottom row: out-of-sample reconstruction RMSE on held-out turning flights. Standard PCA (black) is shown as reference in each panel. Factor Analysis and Sparse PCA recover closely aligned subspaces; Z-score PCA and ICA diverge for understood reasons (see text).*

## 10 Comparing Individual Morphing Shape Modes

### 10.1 Individually Defined Morphing Shape Modes

To compare how individuals vary, we ran PCA on each individual separately and the resulting component loadings were compared with the pooled basis. When the absolute dot products between pooled and per-bird axes were plotted (Figure O), the first several modes formed clear diagonals, showing that the dominant morphing patterns are shared across individuals.  Minor off-diagonal structure reflects small rotations or exchanges of order among neighbouring modes, expected when eigenvalues are similar. The first four modes are therefore show consistency across all hawks, while higher-order modes capture more individualised or context-specific variations.

![[Screenshot 2025-11-04 at 15.56.09.png]]

***Figure O:** Defining the morphing shape modes for each individual and comparing with one hawk, "Toothless" using the absolute dot product values. A diagonal shows perfect alignment.

The first four modes are relatively consistent and share the same information across the different flights (Table G). In the case of "Drogon", the information is more evenly shared across the first two modes, and the 3rd and 4th have swapped definition. 

|**Mode**|**Toothless**|**Rhaegal**|**Ruby**|**Charmander**|**Drogon**|
|---|---|---|---|---|---|
|1|0.997|0.992|0.998|0.996|0.790|
|2|0.993|0.991|0.992|0.978|0.782|
|3|0.628|0.463|0.862|0.449|0.446|
|4|0.651|0.478|0.889|0.400|-0.401|
|5|-0.890|0.462|0.193|-0.831|-0.959|
|6|0.271|-0.453|0.071|0.366|0.610|
|7|0.351|-0.761|-0.043|-0.046|0.324|
|8|0.491|0.530|0.005|0.038|0.296|
|9|-0.705|-0.413|-0.323|0.033|-0.293|
|10|-0.218|-0.967|0.406|-0.555|-0.328|
|11|-0.275|-0.535|0.758|0.342|-0.367|
|12|-0.281|0.585|-0.813|-0.450|-0.546|

***Table G:** Absolute dot-product comparison of pooled versus per-hawk modes.  A diagonal near 1 indicates equivalent axes; off-diagonal structure shows small rotations or swapped order of modes, while sign changes reflect direction reversals of equivalent patterns.

The absolute dot products compare the directions of the PCA axes, and so can exaggerate apparent differences between datasets that do in fact span the same geometric space. Rotations or flipping of PCA axes can occur, and the dot products can be low or negative, even when the underlying morphing shape patterns are the same. To check this, we used a rotation-invariant comparison with the principal cosine metric. For each subspace dimension *k*, we compute the principal cosines between the *k*-dimensional pooled and per-hawk subspaces and report the **minimum** cosine (worst-aligned direction). A value near 1.0 means the *k*-dimensional subspaces are genuinely identical; a drop reveals where they begin to diverge.

> **Note on dimension counting:** Two *k*-dimensional subspaces of ℝⁿ must share at least max(0, 2*k* − *n*) dimensions. With *n* = 12 features, comparing *k* ≤ 6 (= *n*/2) ensures that **no** cosines are trivially forced to 1.0, so every value carries genuine information. The previous analysis (using a single `modes=k` call at *k* = 9) produced trivially guaranteed 1.0 values for the first six modes due to this dimension-counting effect. The sweep approach reported below avoids this artefact.

The principal cosine sweep reveals three regimes (see Notebook 08, principal cosine sweep figure):

| **k** | **Toothless** | **Rhaegal** | **Ruby** | **Charmander** | **Drogon** |
| ----- | ------------- | ----------- | -------- | -------------- | ---------- |
| 1     | 0.996         | 0.990       | 0.994    | 0.992          | 0.807      |
| 2     | 0.995         | 0.998       | 0.994    | 0.972          | 0.985      |
| 3     | 0.669         | 0.493       | 0.841    | 0.526          | 0.247      |
| 4     | 0.954         | 0.885       | 0.795    | 0.809          | 0.907      |
| 5     | 0.895         | 0.602       | 0.050    | 0.745          | 0.906      |
| 6     | 0.179         | 0.828       | 0.455    | 0.204          | 0.430      |

***Table H:** Min principal cosine at each subspace dimension *k* (pooled vs per-hawk). Values near 1.0 indicate identical subspaces. The dip at *k* = 3 followed by recovery at *k* = 4 reveals mode mixing — hawks share a 2D plane (modes 3-4) but orient axes differently within it. Drops at *k* ≥ 5 indicate genuinely individual variation.

**k = 1–2** (min cosine > 0.97 for four hawks): The first two modes — wing lifting and wing spreading — are universal. Drogon is slightly lower at *k* = 1 (0.81) but recovers at *k* = 2 (0.99), confirming the 2D subspace is shared. **k = 3** then **k = 4**: The sharp drop at *k* = 3 (0.25–0.84) followed by recovery at *k* = 4 (0.80–0.95) indicative of mode mixing within a shared 2D plane. **k ≥ 5**: Further drops indicate hawk-specific kinematics.

Permutation testing (200 iterations, sequence-level BirdID shuffling) was used to assess the significance of these alignment patterns. For each permutation, BirdID labels were shuffled at the flight-sequence level (preserving within-flight temporal structure), per-bird PCA was recomputed, and the principal cosine sweep recalculated. The permutation gap — the difference between the observed min cosine and the null expectation — quantifies effect size at each subspace dimension. At *k* = 1–2, the gap was small (≤ 0.015), confirming that alignment is genuinely tight rather than an artefact of high variance concentration. At *k* = 3, the gap widened substantially (0.33–0.75), reflecting the mode-mixing signature. At *k* = 4, the gap recovered to 0.05–0.20, confirming the shared 4D subspace. At *k* ≥ 5, gaps widened further (0.09–0.94), indicating genuinely hawk-specific variation beyond the shared basis.

We also report Tucker congruence coefficients (Lorenzo-Seva and ten Berge 2006) for completeness (Table I). The per-mode Tucker coefficient is equivalent to the absolute dot product (both are cosine similarity on unit eigenvectors), so the per-mode values match Table G. The per-hawk summary (mean across the first 9 modes) gives moderate values (0.46–0.66) because higher-order modes pull down the average. However, dot products and Tucker coefficients are sensitive to axis rotations and order swaps between individuals — two hawks can share the same subspace yet show low dot products if their axes are oriented differently within it. The principal-cosine analysis (Table H) addresses this limitation: it compares subspaces directly (via QR decomposition and SVD) and is invariant to axis rotation within the subspace. This is why the principal cosines recover high alignment at *k* = 4 despite the low dot products at mode 3 — the hawks share a common 4D subspace but orient axes 3 and 4 differently within it.

| **Hawk**   | **Mean matched Tucker** | **Matched coefficients (first 9)**                                |
| ---------- | ----------------------- | ----------------------------------------------------------------- |
| Toothless  | 0.664                   | [0.997, 0.993, 0.628, 0.651, -0.890, 0.271, 0.351, 0.491, -0.705] |
| Rhaegal    | 0.616                   | [0.992, 0.991, 0.463, 0.478, 0.462, -0.453, -0.761, 0.530, -0.413] |
| Ruby       | 0.486                   | [0.998, 0.992, 0.862, 0.889, 0.193, 0.071, -0.043, 0.005, -0.323] |
| Charmander | 0.460                   | [0.996, 0.978, 0.449, 0.400, -0.831, 0.366, -0.046, 0.038, 0.033] |
| Drogon     | 0.545                   | [0.790, 0.782, 0.446, -0.401, -0.959, 0.610, 0.324, 0.296, -0.293] |

***Table I:** Tucker-congruence analysis reveals that the leading morphing-shape components are well matched across birds, while minor modes show greater individual variation.

The first three modes accounted for 87.7-97.4% of variance across individuals, increasing to >99% within nine modes (Table J). This confirms that the morphing shape modes not only generalise across individuals but also retain the key individual-level variation, making them a robust framework for comparing morphing flight.


![[Screenshot 2025-03-14 at 19.54.30.png|500]]
***Figure Q:** Comparing variance explained by splitting the data by individual (colour), by experimental period (1: solid line; 2: dashed), and by obstacle (dotted) and projecting to the nine morphing shape modes. Full numbers provided in Table J.* 

| Hawk       | Exp. | Obst. | Cumulative Variance Explained                                 |
| ---------- | :--: | :---: | :------------------------------------------------------------ |
| Toothless  |  1   |       | 0.658, 0.926, 0.949, 0.971, 0.979, 0.983, 0.988, 0.992, 0.995 |
|            |  2   |       | 0.750, 0.953, 0.970, 0.986, 0.990, 0.991, 0.993, 0.996, 0.997 |
|            |  2   |   ✓   | 0.743, 0.955, 0.974, 0.985, 0.989, 0.991, 0.993, 0.996, 0.998 |
| Drogon     |  1   |       | 0.634, 0.936, 0.951, 0.968, 0.976, 0.981, 0.987, 0.991, 0.994 |
|            |  2   |       | 0.543, 0.927, 0.947, 0.971, 0.979, 0.982, 0.987, 0.991, 0.994 |
|            |  2   |   ✓   | 0.546, 0.884, 0.911, 0.953, 0.964, 0.970, 0.983, 0.990, 0.996 |
| Rhaegal    |  1   |       | 0.690, 0.906, 0.926, 0.950, 0.960, 0.971, 0.981, 0.987, 0.993 |
| Ruby ♀     |  1   |       | 0.679, 0.925, 0.953, 0.970, 0.975, 0.981, 0.985, 0.990, 0.992 |
|            |  2   |       | 0.718, 0.946, 0.965, 0.975, 0.981, 0.985, 0.989, 0.992, 0.995 |
|            |  2   |   ✓   | 0.740, 0.958, 0.972, 0.980, 0.985, 0.988, 0.990, 0.993, 0.996 |
| Charmander |  2   |       | 0.797, 0.939, 0.955, 0.978, 0.985, 0.987, 0.990, 0.994, 0.995 |
|            |  2   |   ✓   | 0.781, 0.935, 0.956, 0.979, 0.984, 0.986, 0.990, 0.994, 0.996 |
***Table J:** Cumulative explained variance for each individual, experimental period, and with and without an obstacle.*


### 10.2 Cross-validation of Shared Modes

To further confirm the generality of the pooled morphing shape mode basis, cross-validation and bootstrapping tests were performed. Two types of cross validation were carried out: Leave-One-Individual-Out (LOIO) and Leave-One-Condition-Out (LOCO). In both cases, the principal components were computed from a subset of the data, excluding all data from the held-out individual or condition, and their ability to reconstruct the excluded data was quantified using a Root Mean Square (RMS) reconstruction error and variance explained. 

LOIO validation assesses whether the shared morphing modes generalise to individuals not included during basis construction. High explained variance (94.4-98.0%) and low RMS errors (0.0109-0.0186) demonstrate that the same geometric coordinates reliably capture morphing across all hawks, even in completely unseen individuals.

LOCO validation further examines robustness with respect to behavioural or environmental conditions, using the same hold-out approach for obstacle conditions, added weight, and perch-perch distances. In all cases, reconstruction errors remained low and variance explained high (typically above 95%), indicating that the primary morphing directions captured by the pooled PCA are robust to changes in context as well as across different birds.

| Validation | Entity                | RMS Error     | Variance Explained (4 modes) |
| ---------- | --------------------- | ------------- | ---------------------------- |
| LOIO       | Each Individual       | 0.0109-0.0186 | 94.4-98.0%                   |
| LOCO       | With/Without Obstacle | 0.0138-0.0173 | 95.3-96.9%                   |
| LOCO       | With/Without Weight   | 0.0150-0.0173 | 95.3-96.6%                   |
| LOCO       | Perch-Perch Distance  | 0.0154-0.0160 | 95.6-96.7%                   |
These complementary analyses demonstrate that the dominant morphing shape modes are shared and stable across all individuals and experimental contexts, capturing nearly all variation in Harris’ hawks’ morphing flight. While minor modes reveal meaningful individual and context-specific refinements, the pooled basis provides a powerful low-dimensional coordinate system facilitating direct comparison and generalisation across individuals and conditions.

### 10.3 Bootstrap Analysis

To determine the stability and effective dimensionality of the shared morphing shape modes basis, a frame-level bootstrap analysis was performed. In each of 2,000 resampling repetitions, all 252,630 non-obstacle training frames were resampled with replacement, a new PCA was fitted, and the resulting modes were used to reconstruct the original training data. This tests whether the morphing shape subspace is sensitive to the particular sample of frames observed. Note that frame-level resampling does not account for temporal autocorrelation within flights (consecutive frames are highly correlated), so the confidence intervals reported below may be narrower than a flight-level or block bootstrap would produce. However, the §10.1 permutation test uses sequence-level shuffling (BirdID labels permuted per flight), providing a complementary test that does respect temporal structure.

These results confirm the rapid convergence in reconstruction error and variance explained with increasing number of modes. Median RMS error decreased from 0.045 with one mode to 0.015 with four modes, while cumulative variance explained rose from 66.7% to 96.5%. All 95% confidence intervals spanned less than ±0.001 for RMS and less than ±0.3% for variance explained, demonstrating the high stability and repeatability of the morphing shape modes.

| Modes (k) | Median RMS | Median Var Explained |
| --------- | ---------- | -------------------- |
| 1         | 0.045      | 0.667                |
| 2         | 0.023      | 0.912                |
| 3         | 0.019      | 0.944                |
| 4         | 0.015      | 0.965                |
| 5         | 0.013      | 0.974                |
| 6         | 0.011      | 0.980                |
| 7         | 0.009      | 0.986                |

Overall, these complementary tests demonstrate that the same low-dimensional morphing geometry underlies all individuals and contexts.  The four leading modes—symmetric flapping, wing spreading, wing sweeping, and tail spreading—account for nearly all variance and generalise across birds, while additional minor components describe small, context-specific refinements.  The shared subspace therefore provides a single geometric coordinate system for comparing morphing flight across individuals and experimental conditions.

## 11 Visualising the Morphing Shape Modes

### 11.1 Projection and Animation

To visualise the eigenvectors from PCA, referred to as morphing shape modes, we projected the complete set of unilateral frames (289,528) onto the new coordinate system defined by the modes (See Supplementary animations). This transformation produces a set of scores which represent how strongly every morphing shape mode contributes to the shape variation in each frame and in which direction. 

To illustrate how each morphing shape mode influences wing and tail shape, we generated a series of synthetic animation frames by systematically adjusting the scores per mode and reconstructing back to the original hawk coordinates (the same method as described in "Projection to Symmetrical Components"). For a given score, the coordinates were plotting as polygons with a fixed head and body polygon -- estimated from scaled photograph references. 

To fully animate the effect of each morphing shape mode on the hawk shape, a range of scores were applied to the mean hawk shape to incrementally shift the wing-tail shape along the given morphing shape mode. We used a range of scores between two standard deviations above and below the mean, visualising how the shape deforms in both positive and negative directions. For simplicity, we show purely bilaterally symmetrical morphing shape changes for each mode. Due to the limitations in minimal markers, the animation is purely for illustrative purposes and does not recreate the real anatomy of the hawk. Future work with higher number of markers that follow a more complete wing shape would improve these results. 

By visualising these PCA modes through animations, we provide intuitive insights into how subtle changes in wing and tail shape impact aerodynamic function and flight stability. These visualisations allow researchers to identify the biomechanical significance of shape variations captured by PCA, linking mathematical abstractions to ecological and behavioural contexts of hawk flight.
### 11.2 Morphing Shape Mode Interpretation

Based on variance explained and their influence on the morphing shape changes, we describe nine morphing shape modes, which contain >99% of the variation. The first four modes are considered major planform shape changes (>96%), modes 5 and 8 describe local incidence changes (1.29% combined), and 6, 7, 9 describe hand-wing shape changes (1.57% combined). Although these latter modes contribute little variance relative to the dominant axes, their influence on aerodynamic control is likely substantial, given that fine-scale hand-wing and incidence adjustments are critical for manoeuvring and stability. Modes 10-12 are treated as noise based on their effect on the shape and very low variance explained (0.68%). 

Real morphing shape changes arise from the coordinated interaction of multiple modes, rather than individual modes in isolation. It is important to note that these visualisations do not necessarily represent discrete control strategies or postures that the hawk explicitly adopts. Instead, each morphing shape mode is a component of the overall morphing space, with real flight behaviour emerging from combinations of multiple modes rather than any single mode in isolation. The scores reflect how much each shape mode contributes to a given moment in flight, but actual wing and tail configurations are governed by their coordinated interaction. 

To illustrate this difference, we compare the distribution of scores for two components across frames (Figure R, right) to the distribution of flight frames that closely resemble each projected score (Figure R, left). The shape similarity is quantified using root-mean-square (RMS) distance between reconstructed shapes and all original dataset shapes, with a threshold determining similarity.  If the score of the first component alone accurately predicted the actual shape of the hawk per frame, we would see a similar histogram on the left as the right. While every score has corresponding real frame, the direct shape predicted from a single morphing shape mode is less apparent. The correspondence is incomplete, confirming that single components alone cannot reproduce the actual flight shapes.

However, when plotted as a 2D histogram (Figure R, bottom row), we observe a close association between score frequency and the corresponding flight shapes in the data. This is expected, as the first two morphing shape modes capture over 90% of the total variance. Yet, their individual contributions alone do not fully reconstruct real morphing flight shapes, reinforcing that flight control relies on the interaction of multiple shape change modes rather than isolated adjustments along a single component. 

![[Screenshot 2025-03-14 at 20.22.01.png|500]]
***Figure R**: Comparison of score distributions and real flight shapes across the first two morphing shape modes. Rows 1 & 2: Histograms of frame counts for shapes closely matching (RMS<0.025) projected scores (left) and frequency distributions of raw scores across all frames (right) for wing lifting (PC1, top) and wing spreading (PC2, middle). Bottom row: the distribution of similar shapes to the combinations of the two modes more closely matches the score frequency, verifying the morphing shape modes only represent meaningful morphing flight shapes in combination.*

The key advantage of decomposing complex morphing into PCA-derived modes is interpretability: each mode is a continuous, quantifiable variable that can be compared directly across individuals, conditions, and behavioural contexts. This makes morphing shape modes a compact yet meaningful basis for analysing and comparing flight kinematics.
### 11.3 Shape Reconstruction Accuracy

To assess how much shape information is retained in the lower-dimensional space, we calculated the reconstruction error of the original marker positions when using only a subset of the principal components. This test quantifies how well the PCA basis captures morphing structure and identifies the number of components beyond which additional variance adds negligible explanatory power.

The reconstruction error was computed as the root mean squared error (RMSE) between the original marker positions and their reconstruction using 1 to 12 principal components. Since all spatial data were normalised by the maximum wingspan of each bird in each year, the resulting RMSE values are expressed as a proportion of wingspan. This allows for direct biological interpretation of the error values across individuals and conditions.

The first principal component alone reduced reconstruction error to 4.5% of wingspan (~4.5-5 cm depending on the individual). With four components, the error fell below 1.5 cm. After six components, the reconstruction error fell to approximately 1.1 cm across all individuals. Beyond eight components, improvements were negligible.

These results confirm that the dominant morphing shape patterns are captured by a low-dimensional subspace and that truncating PCA at 4-6 components retains nearly all relevant shape information with high precision.

## 12 Left and Right Morphing Asymmetry

PCA was performed treating the left and right sides independently, and so we can assess how the hawks use bilateral symmetry or asymmetry in morphing by comparing the left and right scores for every frame. We used major-axis (MA) regression — the first eigenvector of a 2-component PCA on paired left and right scores — which minimises the perpendicular distances to the fitted line and is appropriate when both variables (left and right) contain measurement uncertainty and are conceptually interchangeable.

For each morphing shape mode, if the MA regression slope is close to 1 and the intercept is near 0, this suggests bilaterally symmetric control of that morphing shape change. If the slope deviates from x=y then then the morphing shape is more independently controlled across the left and right sides. The shape modes are not ordered by variance explained; instead, functionally related modes are grouped, with “collective pitching” repositioned for clarity.

As an overall baseline, we first examined left-right symmetry across all non-obstacle flights (131,865 frames from 1,380 flights across all hawks). The major planform modes (1–4) showed MA regression slopes close to 1.0 and intercepts near zero, confirming strong bilateral coupling in the dominant shape changes. Higher modes showed progressively more left-right independence, consistent with fine-scale adjustments being less tightly coupled across the two sides. This baseline provides context for the condition-specific comparisons that follow.

![[S11_symmetry_all_nonobstacle.pdf]]
***Figure S0:** Baseline left-right symmetry across all non-obstacle flights (131,865 frames, 1,380 flights, all hawks). Each panel shows left vs right scores for one morphing mode, with the MA regression fit (dotted line) compared to perfect symmetry (solid grey line, x = y). Major planform modes (1–4) show strong bilateral coupling; higher modes show progressively more independence.*

### 12.1 Symmetry in Modes 1&2

In order to correct for whole body rotations (described in above "Projection and Rotation Analysis"), we assumed left-right symmetry for the first two modes. For clarity, we show results for left-right symmetry from data before rotation correction for these first two modes, the rest are from corrected data. This mixing is valid because the first two modes are identical in definition regardless of whether rotation correction is applied (dot products ≥ 0.999, see Figure N), so pre- and post-correction scores are interchangeable for these modes.

For completeness, we also include below asymmetry results for the unrotated PCA modes (Figure S) in turning flights. Note that with banked turns, yaw and roll will result in false anti-symmetry and asymmetry for the morphing modes. Also note the "looped" appearance of PC1 and 2, which we interpret as being caused by roll and yaw and not asymmetrical morphing in wing lifting and spreading. As such we do not find these results useful for morphing interpretation.  As modes 1 and 2 are identical in definition in the rotated dataset, we use these asymmetry scores in final analysis (Figure T & U). However, the rest of the modes, while related to the morphing shape modes, contain whole body rotations and are more difficult to interpret, and so are not used further. 
![[Fig03c_NOROTATIONAPPLIED_ObstacleExp.png|400]]
***Figure S**: From data including whole body rotation, mode symmetry in obstacle flights by experienced hawks (N=194 flights) for all frames (n=14,921). Roll and yaw have not been removed from the data, while body pitch which has been estimated from the body markers was removed. Modes 1&2 do not contain whole body rotations and are defined the same as those from the corrected data. We do not use other modes for analysis as their interpretation is challenging. As these modes contain roll and yaw, anti-symmetry is visible in many modes.*

### 12.2 Symmetry during Flapping and Gliding

In flights by experienced hawks flying 9m and 12m with no obstacle, we can compare symmetry during flapping acceleration (n=11,218 frames, N=233 flights by 4 hawks) and gliding deceleration (n=15,840 frames, N=229 flights) (Figure T). We exclude 5m and 7m as they do not contain a long gliding phase. Scores for Modes 1 & 2 calculated from unrotated data, while Modes 3-9 are calculated from corrected data. 


![[symmetry_scores 2.png|700]]

***Figure T: Symmetrical and asymmetrical shape changes in morphing.** Many of the shape changes in morphing flight are highly symmetrical, indicative of bilateral coupling. During deceleration, hand-wing adjustments and tail lowering (collective pitch) show more left-right independence. Outer images: Nine morphing shape modes projected symmetrically in both extremes (±2 standard deviations). Wing and tail movements within a mode are highly correlated (see Supplementary Videos). Bottom left subset shows marker placement on hawks.  Symmetry comparisons between the right-side and left-side shape changes (left vs right PCA scores per frame). Variance explained (%) by major-axis regression equation (dotted line) relative to perfect symmetry (solid line, x=y). In 9m and 12m flights by experienced birds during flapping acceleration (left columns; n=11,218 frames, N=233 flights by 4 hawks) and gliding with landing (right columns, n=15,840, N=229, 4 hawks). 

Even in flights without an obstacle, the hawks often flew using curved trajectories between perches, meaning some asymmetric control is expected (see Figure E, first column). However, because these turns varied across flights, their effects should average out, allowing us to determine which morphing shape modes are consistently symmetric or asymmetric across conditions. In obstacle flights, this analysis helps identify which asymmetric morphing strategies could contribute to turning from four individuals.

### 12.3 Morphing Symmetry in different Flight Contexts

Additionally we can compare symmetry during acceleration and deceleration with and without an obstacle and by juveniles versus experienced hawks (Figure U). As before, the first two modes scores have been calculated with unrotated data and so "looping" is likely more indicative of banking than morphing asymmetry. We see juveniles use more left-right independent morphing. In flights with an obstacle, the hawks used banked turns (see whole body angle results). Whole body rotation in manoeuvres redirect aerodynamic forces (Ros et al 2011), in addition to imbalances produced by intrinsic shape changes relative to the body. 

Before an obstacle, the hawks pitched their wings more negatively than in straight flights, while the tail was more spread and more asymmetric (e.g. tail banking). After passing the obstacle, yaw stabilised while roll decreased in order to steer towards the perch, with hawks also executing the rapid pitch-up manoeuvre for landing (Carruthers 2007). In this phase, morphing post-obstacle showed greater asymmetry for hand-wing shape changes and tail spreading, while collective pitching showed anti-symmetry, indicating tail banking. 

![[symmetry_conditions.png]]

***Figure U:** Symmetry scores for different experimental conditions in acceleration and deceleration. Overall, deceleration showed more asymmetry, as did juveniles in comparison to experienced hawks. There were also more asymmetry after navigating the obstacle. For frame and flight counts see Table B.* 
## 13 Time-trace Morphing Scores

To visualise the temporal variation of each morphing shape mode across flights, we calculated the binned mean and standard deviation for the scores ($S$) from all birds. This allowed us to capture the overall trends across all individuals. As before, “collective pitching” (originally PC08) has been repositioned after PC05 (counter pitching) to group the two pitching-related modes together. The display order is therefore: PC01, PC02, PC03, PC04, PC05, PC08, PC06, PC07, PC09 — and so these are not strictly ordered by variance explained. The scores for each component are plotted as binned mean scores (bin size = 0.1m) across all the hawks with shaded ±1 standard deviation around the mean. 

### 13.1 Morphing During Varying Flight Distances

We also show the scores for different perch-perch distances. Data taken from experimental period 1, with no obstacle or added weights, from one experienced female and three juvenile males (Figure V). The flapping wingbeat cycles are visible in the first few metres of flight, with flapping amplitude decreasing with each successive wingbeat but span increases (modes 1 & 2). For the longer distances (9m & 12m) the hawks transition to a glide with maximum spread and steadily lowering wings to neutral. In the shorter flights (5m & 7m), wingspan steadily increases over the flap cycles that approach landing. A small peak in wing lifting occurs after the glide in the 1m before landing. 


![[2025-03-15_AllScoresPerchDist 1.png]]
***Figure V:** Comparing scores during flights of different distances, where perches are 5, 7, 9, 12m apart. The binned mean (0.1m bin size) for all hawks is shown with ±1 standard deviation shaded. Variation between individuals and within individuals is apparent for the smaller modes.* 


The wings sweep oscillates during the wingbeat cycles (mode 3). Sweep differs in the 9m glides compared with 12m glides: oscillations of forward and backwards sweep continue in the shorter glide while in the longer glide the transition to gliding the wings are more swept back and then during the glide wings steadily sweep forwards. In all perch-perch distances, a peak of positive (forward) sweep occurs around 1.5m before the perch. 

Tail spreading (mode 4) oscillates with wing beat cycles and increases on approach to the perch. At the point of landing in all flight distances, tail spreading reaches maximum and decreases with foot contact (approx -0.3m before the perch). At the transition to gliding in the longer flights, tail spread is minimal and then increases over the glide. Tail spreading coincides with the hawks pitching up (see "Projection to Symmetrical Components", Figure K). 

Counter pitching (mode 5) describes the tail lowering while the wings pitch in the opposite direction but with smaller magnitude. During flapping, wing pitch oscillates, but is on average negative (the trailing edge above the leading edge). In gliding, it is increases and at max, seen at 1.5m before the perch, the leading and trailing edge are the same height in the body coordinate space.  
There is a sudden reduction in wing pitch around 1m before the perch, during the rapid whole body pitch-up. This coincides with the wings sweeping forwards briefly. In combination, these shape changes resemble what we term a "shrug" — a rapid, coordinated forward sweep and pitch reduction of the wings during the final pitch-up before landing. This manoeuvre is visible in our high-speed video recordings of the hawks and is consistent with the landing sequence described by Carruthers et al (2007) in a Steppe eagle. In that footage and in ours, the covert feathers are visibly deflected upward under stall conditions immediately prior to the shrug, then flatten abruptly afterwards. This sequence suggests the shrug could function as a flow-reattachment mechanism: the sudden forward sweep and pitch change may re-energise the boundary layer over the upper wing surface, recovering attached flow after a transient stall during the pitch-up. However, confirming this aerodynamic interpretation would require pressure or flow-visualisation measurements, which are beyond the scope of this study.

### 13.2 Obstacle versus Control Flights

Data taken from Experimental Period 2, with and without an obstacle. We can plot the scores with and without an obstacle (see Main Text: Figure 2). Between individuals their wingbeat frequency differs which affects the means. Comparing across flights and individuals with different wingbeat frequencies would require time-series decomposition methods (e.g. dynamic mode decomposition, wavelet analysis) that are not explored here. 

A sudden increase and decrease in wing sweep is visible around the obstacle at the midpoint between the perches. There is also a new, highly variable, decrease in collective pitching (where the tail drops as the wings pitch up). A decrease and increase in hand-wing sweep is also visible around the obstacle.

![[2025-02-22_Scores.png]]
***Figure X:** Comparing scores during flights with and without an obstacle, where perches are 9m apart. Hawks flew with and without added weight in the form of an IMU. Control flights, N=256 flights, 29949 frames; obstacle flights N=254, 18426 frames. The binned mean (0.1m bin size) for all hawks is shown with ±1 standard deviation shaded.  

### 13.3 Mean Scores by Individuals under Different Contexts

Individual variation is high in both morphing technique and the timing of manoeuvres. For completeness, we show each individual with no obstacle and no weight; with obstacle and no weight; and with both an obstacle and a weight (Figure Y). The standard deviation is greatly reduced compared with the pooled data, showing that morphing is more consistent within individuals than across them.

Several inter-hawk differences are visible. Hawks differ in their wingbeat frequency, which shifts the spatial position of the flap-to-glide transition: for example, Toothless transitions to gliding earlier (further from the perch) than Drogon in control flights. Drogon shows higher-amplitude wing sweep oscillations during flapping (mode 3) and a more pronounced tail-spread increase during the glide (mode 4). In obstacle flights, the timing of the obstacle-related sweep adjustment varies by individual, with some hawks initiating the manoeuvre earlier. The added-weight condition generally increases flapping amplitude in mode 1 (wing lifting) across all hawks, though the effect is most pronounced in Charmander. These individual differences are consistent with the principal-cosine analysis (§10) showing shared modes but individual modulation.

![[2025-03-15_ToothlessScores 1.png]]

![[2025-03-15_DrogonScores 1.png]]

![[2025-03-15_RubyScores 1.png]]

![[2025-03-15_CharmanderScores 1.png]]
***Figure Y:** Comparing scores during flights under different experimental conditions and by individual. Perches are 9m apart. Left column: control flights; middle column: obstacle flights with no added weight; right column: obstacle flights with added weight in the form of a worn IMU. Rows show individual hawks, each subplot shows a morphing shape mode score. The binned mean (0.1m bin size) shown with ±1 standard deviation shaded. For flight counts and frame counts see Table B.

### 12.4 Naive versus Experienced Hawks

Three hawks (Drogon, Toothless, Rhaegal) were recorded both as naive juveniles (2017, experimental period 1) and as experienced adults (2020, experimental period 2), providing a within-individual developmental comparison. For each hawk, score traces from 9 m straight flights (no obstacle, no added weight) were compared between periods. Naive flights are shown as dotted lines and experienced flights as solid lines, with ±1 standard deviation shaded.

Juvenile hawks used qualitatively similar morphing patterns to their experienced counterparts — the same sequence of flapping, gliding, and landing manoeuvres is visible in both periods. However, the naive score traces showed greater variability (wider standard deviation bands), suggesting less consistent morphing control. This is consistent with the expectation that flight motor patterns become more refined with practice. The major planform modes (1–4) showed the clearest developmental consistency, while higher modes showed more individual variation between periods.

![[S12_scores_naive_experienced_PC01.pdf]]
![[S12_scores_naive_experienced_PC02.pdf]]
![[S12_scores_naive_experienced_PC03.pdf]]
![[S12_scores_naive_experienced_PC04.pdf]]
***Figure Y2:** Score traces comparing naive juvenile (2017, dotted) and experienced adult (2020, solid) flights for three hawks (Drogon, Toothless, Rhaegal) on 9 m straight flights. Binned mean (0.1 m bins) with ±1 standard deviation shaded. Juvenile flights show qualitatively similar patterns but with greater variability.*

## 14 Evidence for  Flight Behaviour on a Continuum

We tested whether flight shapes form discrete clusters or lie on a continuum within the morphing shape space (Figure Z). Using multiple complementary unsupervised methods, we find consistent evidence that the data do not separate into distinct groups.

Gaussian mixture models (GMMs) were fitted using the first 6 PCA dimensions in a two-stage approach: a fast diagonal-covariance sweep (k = 1–50 on a 50,000-frame subsample), followed by full-covariance refits of the top candidates on all 289,528 frames. The BIC decreased monotonically without reaching a minimum at any small k; instead, the best fit occurred at k = 50, requiring roughly 50 equally-sized Gaussian clusters to approximate the density. All clusters remained active and balanced (each capturing 3.6–4.8% of frames), with silhouette scores near zero (0.006 at best). This indicates the GMM tiles a continuous density rather than identifying discrete behavioural modes.

Silhouette analysis across k = 2–10 and 4–12 PCA dimensions confirmed uniformly weak cluster separability (coefficients of 0.13–0.24 at k = 10), well below the ~0.5 threshold for distinct groups. DBSCAN (ε = 0.3–1.0, `min_samples=10`, 50,000-frame subsample across k = 4, 6, 9, 12 PCs) consistently found a single connected cluster with 0% noise, reinforcing that the shape distributions form one cohesive manifold.

The Levina–Bickel nearest-neighbour method (Levina and Bickel 2004) estimated intrinsic dimensionality at ~3.5–4.6 across neighbourhood sizes (m = 5–60), consistent with a manifold of approximately four to five effective dimensions. This agrees with the four leading PCA modes explaining 96.3% of variance.

Taken together, all methods converge on the same conclusion: hawk flight behaviours occupy a continuous, low-dimensional manifold. Rather than switching between discrete states or gaits, the hawks modulate their wing and tail configurations smoothly within a shared morphing space.

![[S13_continuum_summary.pdf]]
***Figure Z:** Continuum analysis summary. (a) BIC curve showing monotonic decrease with no minimum at small k, with an elbow at k ≈ 10 marking the transition from coarse density capture to refinement. (b) Score space coloured by GMM cluster assignment (k ≈ 50), showing smooth tiling rather than isolated clusters. (c) Levina–Bickel intrinsic dimensionality estimates converging on ~4–5 effective dimensions.*

## S14 Experimental Effects on Morphing Flight

*Generated from notebook 13_ExperimentalEffects.ipynb*

In addition, we used heatmaps (Main text: Figure 4) to visualise more detailed temporal and fine-scale changes in the morphing shape modes across experimental conditions. Each morphing shape mode score is represented as a row in the heat maps, with experimental conditions stacked for comparison. The heat maps were scaled to the 1st and 99th percentiles of each morphing shape mode's score distribution, with blue representing negative scores, red representing positive scores, and yellow indicating values near the mean hawk shape. 

The heatmaps were scaled to the 1st and 99th percentiles rather than the full range to reduce the visual influence of rare extreme scores whilst preserving the dynamic range of typical morphing variation. This provides a closer look at individual flight behaviour for consistent behaviours. We include these heatmaps for transparency: as within- and across-individual variation is high, they do not represent stereotyped generalisable behaviours and are not suited for direct comparison across individuals with different wingbeat frequencies. More advanced time-series methods that can align and decompose flights of different durations and frequencies would be required for such comparisons.

### Obstacle Flights

We compared control and obstacle flights for Toothless and Ruby to examine flight responses to environmental challenges. Heatmaps display the upper row as the obstacle condition and the lower row as the control condition. A vertical line marks the obstacle position at −4.5 m from the perch. Flights with added IMU weight were excluded from this analysis.

Colour coding follows the convention described above: blue indicates negative scores (wings lowered, contracted, or swept back), red indicates positive scores (wings raised, spread, or swept forward), and yellow represents values near the mean hawk shape.

Key findings emerge from the score analysis:

**Wing spreading (PC02)** exhibits a pronounced negative dip at 6.5–4.5 m in both hawks (Toothless −0.017, Ruby −0.036), indicating that wings fold immediately before the obstacle, then recover afterwards. This pattern suggests a span-reduction manoeuvre deployed for gap negotiation through confined spaces.

**Handwing spreading (PC06)** increases consistently in both hawks across all distance bins (ranging +0.003 to +0.010), indicating that whilst overall span is reduced, the handwing spreads to maintain control authority and aerodynamic responsiveness.

**Tail spreading (PC04)** remains consistently positive in Toothless (ranging +0.007 to +0.015) and shows a weaker but similar pattern in Ruby. Concurrent tail spreading with minor wing sweep-back during obstacle flights suggests coordinated control of multiple effectors.

**M-folding (PC07)** shifts markedly negative (towards w-folding) in both hawks across all distance bins—not merely localised at the obstacle. This suggests a globally altered wing configuration during obstacle flights rather than a purely localised adjustment at the critical point.

The response pattern was qualitatively similar between the two hawks, with both recruiting the same morphing modes, though magnitudes differed: Ruby showed a stronger response in PC02, whilst Toothless exhibited a stronger response in PC04.

![Heatmap: obstacle condition versus control for Toothless. Upper row = obstacle flight; lower row = control flight. Vertical line marks obstacle position at −4.5 m.](figures/supplementary/S14_obstacle_Toothless.pdf){width=90%}

![Heatmap: obstacle condition versus control for Ruby. Upper row = obstacle flight; lower row = control flight. Vertical line marks obstacle position at −4.5 m.](figures/supplementary/S14_obstacle_Ruby.pdf){width=90%}

### Weight Effect

We compared flights with and without added IMU weight for Charmander and Ruby to examine how added dorsal load alters morphing flight dynamics. Heatmaps display the upper row as the weighted condition and the lower row as the control condition.

**Wing sweeping (PC03)** shifts markedly positive (wings swept forwards) in both hawks, with the effect strongest near landing (Ruby +0.027 at 2.5–0.5 m). Forward wing sweep under load is consistent with aerodynamic compensation: shifting the centre of pressure anteriorly to counterbalance the added dorsal mass.

**M-folding (PC07)** shifts negative (towards w-folding) in most distance bins for both hawks, mirroring the pattern observed in obstacle comparisons. This suggests a general aerodynamic load-compensation response rather than an obstacle-specific reaction.

**Counter pitching (PC05)** shows weakly negative values in both hawks, indicating that the leading edge pitches downward under weight. However, magnitudes remain small (≤0.024), suggesting this is a minor adjustment.

Weight effects are substantially smaller than obstacle effects (largest magnitude ~0.03 versus ~0.04 for obstacles). The most robust and consistent finding is the forward wing sweep (PC03), which appears reliably across both hawks and all distance bins.

![Heatmap: weight condition versus control for Charmander. Upper row = weighted flight; lower row = control flight.](figures/supplementary/S14_weight_Charmander.pdf){width=90%}

![Heatmap: weight condition versus control for Ruby. Upper row = weighted flight; lower row = control flight.](figures/supplementary/S14_weight_Ruby.pdf){width=90%}

### Experience and Maturation

Drogon and Toothless were recorded as both naive juveniles in 2017 (Period 1) and as experienced adults in 2020 (Period 2), permitting a longitudinal within-individual comparison of how flight morphing changes with experience and maturation.

**Wing sweeping (PC03)** exhibits the clearest developmental signal: experienced hawks sweep their wings markedly backwards across the entire approach phase, with Drogon ranging −0.05 to −0.11 and Toothless ranging −0.05 to −0.08. By contrast, naive juveniles held their wings near neutral or displayed only slight forward sweep. This shift towards retracted wing postures is consistent with higher approach speeds and more aggressive braking dynamics typical of experienced perching.

**Tail spreading (PC04)** decreases substantially with experience in both hawks across all distance bins (−0.02 to −0.05), indicating that experienced birds contract their tail and position it more anteriorly. This contributes to a more streamlined, retracted flight profile.

**Wing spreading (PC02)** becomes consistently negative in three of four distance bins for both hawks with experience, indicating a more folded overall wing posture that complements the backward sweep.

**Collective pitching (PC08)** shifts positive in most bins for both hawks in the experienced state, representing a trailing-edge-up pitch adjustment that alters body trim.

Experience produces the largest shifts in morphing dynamics observed across all experimental comparisons—approximately three to five times larger than the magnitude of obstacle or weight effects. The collective pattern (backward wing sweep combined with wing folding and tail contraction) produces a more retracted, streamlined flight profile entirely consistent with higher approach speeds and more forceful braking manoeuvres deployed by experienced birds.

![Heatmap: experience effect for Drogon. Comparison of naive juvenile (2017, upper row) versus experienced adult (2020, lower row) flights.](figures/supplementary/S14_experience_Drogon.pdf){width=90%}

![Heatmap: experience effect for Toothless. Comparison of naive juvenile (2017, upper row) versus experienced adult (2020, lower row) flights.](figures/supplementary/S14_experience_Toothless.pdf){width=90%}

### Individual Comparisons

Two independent individual comparisons were conducted: (a) Drogon versus Rhaegal (siblings, both naive in 2017), and (b) Toothless versus Ruby (both experienced in 2020).

**Siblings (Drogon versus Rhaegal):**

Handwing spreading (PC06) reveals a stable individual signature: Rhaegal remains consistently more contracted across all distance bins (~−0.02), suggesting a fixed morphological or motor trait. Wing spreading (PC02) differs in the early flight phase (Rhaegal more spread at 8.5–4.5 m) but reverses near landing—a distance-phased difference rather than a uniform individual signature.

**Experienced individuals (Toothless versus Ruby):**

M-folding (PC07) exhibits the clearest individual signature: Ruby shows consistently more w-folding across all distance bins (~−0.019), with a remarkably uniform magnitude. This striking consistency suggests a stable individual habit or morphological constraint.

Counter pitching (PC05) is consistently more negative in Ruby across all distance bins (−0.013 to −0.026), indicating that Ruby flies with its leading edge pitched further downward—a trim adjustment that may reflect individual morphology or habitual motor strategy.

Tail spreading (PC04) is consistently more positive in Ruby across all distance bins, indicating that Ruby spreads its tail more extensively than Toothless, again suggesting a stable individual signature.

Modes that distinguish individuals (PC05, PC06, PC07) are geometrically finer adjustments: handwing conformation, chordwise camber, and pitching trim. These likely reflect individual morphology or habitual motor strategies developed through experience. By contrast, task-driven modes (PC02, PC03) vary strongly with distance throughout the approach but do not show consistent differences between individuals, suggesting they are modulated by immediate aerodynamic demands rather than fixed individual traits.

![Heatmap: individual comparison between siblings. Comparison of Drogon versus Rhaegal (both naive, 2017).](figures/supplementary/S14_siblings_Drogon_Rhaegal.pdf){width=90%}

![Heatmap: individual comparison between experienced adults. Comparison of Toothless versus Ruby (both experienced, 2020).](figures/supplementary/S14_individuals_Toothless_Ruby.pdf){width=90%}

### Summary Table

The dominant morphing shape modes, effect magnitudes, and spatial patterns vary systematically across experimental comparisons:

| Comparison | Dominant modes | Effect size | Spatial pattern |
|---|---|---|---|
| Obstacle | PC02, PC04, PC06, PC07 | Moderate (~0.02–0.04) | PC02 localised at obstacle; others global |
| Weight | PC03, PC07, PC05 | Small (~0.01–0.03) | Global, strongest near landing |
| Experience | PC03, PC04, PC02, PC08 | Large (~0.05–0.11) | Global across full approach |
| Individual | PC05, PC06, PC07 | Moderate (~0.02) | Uniform—distance-independent |

Obstacle and weight effects emerge as localised or task-driven responses to immediate environmental perturbations. Experience effects are substantially larger and span the entire approach, reflecting fundamental changes in flight strategy and motor control. Individual modes are uniformly expressed across distance, indicating stable morphological or habitual traits.

## 15 Individual Comparisons

In addition, we used heatmaps (Main text: Figure 4) to visualise more detailed temporal and fine-scale changes in the morphing shape modes across experimental conditions. Each morphing shape mode score is represented as a row in the heat maps, with experimental conditions stacked for comparison. The heat maps were scaled to the 1st and 99th percentiles of each morphing shape mode’s score distribution, with blue representing negative scores, red representing positive scores, and yellow indicating values near the mean hawk shape. 

The heatmaps were scaled to the 1st and 99th percentiles rather than the full range to reduce the visual influence of rare extreme scores while preserving the dynamic range of typical morphing variation. This provides a closer look at individual flight behaviour for consistent behaviours. We include these heatmaps for transparency: as within- and across-individual variation is high, they do not represent stereotyped generalisable behaviours and are not suited for direct comparison across individuals with different wingbeat frequencies. More advanced time-series methods that can align and decompose flights of different durations and frequencies would be required for such comparisons.

## 16 Generalising across species

In this analysis we have used components calculated from all data pooled and projected onto individuals such that each is comparable and within the same morphing shape change space. As PCA provides linear components, while there are disadvantages for non-linear data, the advantage is explainability and providing a common coordinate system for direct comparison. There is also the potential to expand common coordinate system to different morphologies, such as different species of birds. 

To evaluate the feasibility of generalising extracted morphing shape modes as a shared coordinate system for comparative flight analysis, we projected these modes onto cadaver-based morphological reconstructions of 21 avian species with distinct wing and tail morphologies. No flight kinematics data were available for these species, and this analysis does not aim to reconstruct their actual flight behaviour. Instead, this projection serves as a proof of concept that morphing shape modes provide a shared framework for comparing morphing flight across species, accommodating morphological differences. 

### 16.1 Adjusting Cadaver Measurements

Cadaver measurements were taken with permission from published data (Harvey et al 2022b), with 4 landmark positions per side chosen to correspond as closely as possible to the Harris' hawk marker locations (wingtip, primary, secondary, tailtip), plus 3 rigid positions used only for animations (head, tail root, and shoulders) — 11 landmarks total per species. Not all cadaver morphologies map cleanly onto the hawk marker layout; in particular, species with very different wing planforms (e.g. swifts, pheasants) required approximate correspondence. We chose cadaver measurements at the point of maximum wingspan. We note the cadaver tails were furled, unlike their functional position in flight, which uses a spread tail particularly during slow manoeuvring flight as was recorded with the hawks. To approximate its role in flight, we pragmatically doubled the tail width in the cadaver measurements. This is not derived from any formal source and is a simplification for the proof-of-concept demonstration. A central reference point equivalent to the backpack marker (mid-shoulders) was used as the origin.

| Species Common Name    | Wing Span (cm) |
| ---------------------- | -------------- |
| American white pelican | 195.0          |
| Amherst pheasant       | 64.6           |
| Barn owl               | 106.0          |
| Belted kingfisher      | 56.1           |
| Black swift            | 38.5           |
| Canada goose           | 156.0          |
| Common nighthawk       | 52.0           |
| Common raven           | 116.0          |
| Coopers hawk           | 81.2           |
| Glaucous winged gull   | 132.4          |
| Great blue heron       | 184.4          |
| Himalayan monal        | 84.2           |
| Mallard                | 88.8           |
| Merlin                 | 83.2           |
| Northern flicker       | 51.3           |
| Peregrine falcon       | 83.9           |
| Pigeon                 | 63.4           |
| Sharp shinned hawk     | 61.0           |
| Silver pheasant        | 76.4           |
| Storm petrel           | 45.2           |
| Western grebe          | 78.6           |
***Table K:** Summary wingspans from different cadavers with their common species name. Taken with permission from published data (Harvey et al 2022b).*

![[Screenshot 2025-03-15 at 14.55.43.png|300]] ![[Screenshot 2025-03-15 at 15.14.05.png|300]]



![[Screenshot 2025-03-15 at 15.12.58.png|300]]![[Screenshot 2025-03-15 at 15.06.53.png|300]]



![[Screenshot 2025-03-15 at 15.10.15.png|400]]![[Screenshot 2025-03-15 at 15.15.43.png|300]]
**Figure Z:** Comparison of wing and tail shape between the Harris' hawk shape (red) and cadaver measurements (light blue). Top row: Amherst pheasant, American White Pelican. Middle row: Peregrine falcon, Steller's jay. Bottom row: Canadian goose, mallard. 

### 16.2 Transformation to Different Morphologies

We used piecewise linear transformations to transform the Harris’ hawk mean shape to each species while preserving local anatomical correspondences. The transformation is "piecewise" in the sense that each marker undergoes its own independent linear transformation (scaling and rotation), rather than applying a single global transformation to the entire shape. For a given marker, the transformation matrix was computed as:
$$T_i = s_i R_i,$$Where $s_i$ is a scaling factor and $R_i$ is a  Rodrigues rotation matrix. The scaling factor was defined as the ratio of the target and source marker norms: 
$$
s_i = \frac{\|\mathbf{m}_i^{(\text{target})}\|}{\|\mathbf{m}_i^{(\text{hawk})}\|}
$$
This ensured that the marker retained its relative length. The rotation matrix was derived from the axis-angle representation, where the rotation axis was determined by the cross product of the source and target markers, and the rotation angle $\theta_i$ was given by:
$$
\theta_i = \cos^{-1} \left( \frac{\mathbf{m}_i^{(\text{hawk})} \cdot \mathbf{m}_i^{(\text{target})}}{\|\mathbf{m}_i^{(\text{hawk})}\| \|\mathbf{m}_i^{(\text{target})}\|} \right)
$$

The rotation matrix $R_i$ was constructed using the Rodrigues’ rotation formula. In the degenerate case where source and target vectors are parallel or anti-parallel (cross product near zero), the rotation axis is undefined. We handle this by selecting the canonical axis (x, y, or z) least aligned with the source vector and computing a perpendicular rotation axis from its cross product with the source. The rotation angle (0° for parallel, 180° for anti-parallel) is then applied around this axis.

Since transformations were computed independently for each marker, we assembled a block-diagonal transformation matrix $T$, where each block corresponded to an individual marker’s transformation. This allowed us to apply marker-wise transformations collectively while maintaining the anatomical integrity of the overall shape. The transformed markers were obtained by applying $T$ to the hawk’s original marker positions as:
$$
M^{(\text{transformed})} = T M^{(\text{hawk})}
$$

After transformation, the tail-tip z-coordinate was overridden to −0.05 m. Cadaver-derived tail positions tend to droop downwards relative to live birds in flight; this manual correction sets the tail tip to a biologically plausible elevation. The value was chosen by visual inspection of the animated result.
### 16.3 Transformation of Morphing Shape Modes 

We applied the same transformation to the principal components derived from hawk flight to maintain the relative morphing shape patterns while adapting them to the new morphology. The transformed principal components were obtained as 
$$P^{(\text{transformed})} = T P$$Since $T$ is not necessarily orthonormal (each marker has a different scale and rotation), the transformed principal components are no longer strictly orthogonal. This means the modes should be interpreted in the original hawk space where they retain their statistical properties; in the transformed species space they serve only as a visualisation and projection tool, not as independent axes.

To demonstrate this functionality, we projected the hawk’s principal component scores onto different species and animated each “flying like a hawk.” The species-specific scores were computed using the transformed principal components with the Moore-Penrose pseudoinverse, ensuring that the morphing shape modes retained the correct amplitudes in the new space. 

Qualitative inspection of the resulting animations (see Supplementary Videos) shows flights that appear biologically plausible based on comparison with video footage of the respective species, though this assessment is subjective and no quantitative validation against real flight data was performed. The results show proof of concept for a shared morphing coordinate space across species. We found highly decorative tail feathers were not well represented by hawk tail measurements, but these differences would be easily compared in the same coordinate space. 

This framework enables future studies to map different morphologies into a shared morphing shape space, facilitating quantitative analysis of shape variation across species. Beyond avian flight, this method could be applied to broader comparative biomechanics, bioinspired robotics, evolutionary biology, and the animation industry, potentially providing a robust approach for interpreting complex motion in morphologically diverse organisms.

## 17 Bibliography

Björck, Åke, and Gene H. Golub. ‘Numerical Methods for Computing Angles between Linear Subspaces’. _Mathematics of Computation_ 27, no. 123 (1973): 579–94. [https://doi.org/10.1090/S0025-5718-1973-0348991-3](https://doi.org/10.1090/S0025-5718-1973-0348991-3).

Björklund, Mats. ‘Be Careful with Your Principal Components’. _Evolution_ 73, no. 10 (2019): 2151–58. [https://doi.org/10.1111/evo.13835](https://doi.org/10.1111/evo.13835).

Carruthers, Anna C., Adrian L. R. Thomas, and Graham K. Taylor. ‘Automatic Aeroelastic Devices in the Wings of a Steppe Eagle *Aquila Nipalensis*’. _Journal of Experimental Biology_ 210, no. 23 (December 2007): 4136–49. [https://doi.org/10.1242/jeb.011197](https://doi.org/10.1242/jeb.011197).

Daffertshofer, Andreas, Claudine J. C. Lamoth, Onno G. Meijer, and Peter J. Beek. ‘PCA in Studying Coordination and Variability: A Tutorial’. _Clinical Biomechanics_ 19, no. 4 (2004): 415–28. [https://doi.org/10.1016/j.clinbiomech.2004.01.005](https://doi.org/10.1016/j.clinbiomech.2004.01.005).

Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. ‘A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise’. In _Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)_, 226–31. AAAI Press, 1996.

France, Lydia A. _Mechanics and Control of Perching Flight_. DPhil thesis, University of Oxford, 2019. [https://doi.org/10.5287/ora-7jadqemxp](https://doi.org/10.5287/ora-7jadqemxp).

Gillies, James A., Adrian L. R. Thomas, and Graham K. Taylor. ‘Soaring and Manoeuvring Flight of a Steppe Eagle Aquila Nipalensis’. _Journal of Avian Biology_ 42, no. 5 (2011): 377–86. [https://doi.org/10.1111/j.1600-048X.2011.05105.x](https://doi.org/10.1111/j.1600-048X.2011.05105.x).

Harvey, Christina, Vikram B. Baliga, Jasmin C. M. Wong, Doug L. Altshuler, and Daniel J. Inman. ‘Birds Can Transition between Stable and Unstable States via Wing Morphing’. _Nature_ 603, no. 7902 (March 2022a): 648–53. [https://doi.org/10.1038/s41586-022-04477-8](https://doi.org/10.1038/s41586-022-04477-8).

Harvey, Christina, Lawren L. Gamble, Christian R. Bolander, Douglas F. Hunsaker, James J. Joo, and Daniel J. Inman. ‘A Review of Avian-Inspired Morphing for UAV Flight Control’. _Progress in Aerospace Sciences_ 132 (July 2022b): 100825. [https://doi.org/10.1016/j.paerosci.2022.100825](https://doi.org/10.1016/j.paerosci.2022.100825).

Kabsch, Wolfgang. ‘A Solution for the Best Rotation to Relate Two Sets of Vectors’. _Acta Crystallographica Section A_ 32, no. 5 (September 1976): 922–23. [https://doi.org/10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873).

KleinHeerenbrink, Marco, Lydia A. France, Caroline H. Brighton, and Graham K. Taylor. ‘Optimization of Avian Perching Manoeuvres’. _Nature_ 607, no. 7917 (July 2022): 91–96. [https://doi.org/10.1038/s41586-022-04861-4](https://doi.org/10.1038/s41586-022-04861-4).

Levina, Elizaveta, and Peter J. Bickel. ‘Maximum Likelihood Estimation of Intrinsic Dimension’. In _Advances in Neural Information Processing Systems_ 17, 777–84. MIT Press, 2004.

Lorenzo-Seva, Urbano, and Jos M. F. ten Berge. ‘Tucker’s Congruence Coefficient as a Meaningful Index of Factor Similarity’. _Methodology_ 2, no. 2 (2006): 57–64. [https://doi.org/10.1027/1614-2241.2.2.57](https://doi.org/10.1027/1614-2241.2.2.57).

Miñano, Sofía, Stuart Golodetz, Tommaso Cavallari, and Graham K. Taylor. ‘Through Hawks’ Eyes: Synthetically Reconstructing the Visual Field of a Bird in Flight’. _International Journal of Computer Vision_ 131, no. 6 (2023): 1497–1531. [https://doi.org/10.1007/s11263-022-01733-2](https://doi.org/10.1007/s11263-022-01733-2).

Ros, Ivo G., Lori C. Bassman, Marc A. Badger, Alyssa N. Pierson, and Andrew A. Biewener. ‘Pigeons Steer like Helicopters and Generate Down- and Upstroke Lift during Low Speed Turns’. _Proceedings of the National Academy of Sciences_ 108, no. 50 (13 December 2011): 19990–95. [https://doi.org/10.1073/pnas.1107519108](https://doi.org/10.1073/pnas.1107519108).

Rousseeuw, Peter J. ‘Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis’. _Journal of Computational and Applied Mathematics_ 20 (1987): 53–65. [https://doi.org/10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7).

Schwarz, Gideon. ‘Estimating the Dimension of a Model’. _The Annals of Statistics_ 6, no. 2 (1978): 461–64. [https://doi.org/10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136).

Shelton, James. _Visually Guided Pursuit in Harris’s Hawks_. DPhil thesis, University of Oxford, 2022. [https://doi.org/10.5287/ora-7evvdjk08](https://doi.org/10.5287/ora-7evvdjk08).

Tomotani, Barbara M., Wender Bil, Henk P. van der Jeugd, Remco P. M. Pieters, and Florian T. Muijres. ‘Carrying a Logger Reduces Escape Flight Speed in a Passerine Bird, but Relative Logger Mass May Be a Misleading Measure of This Flight Performance Detriment’. _Methods in Ecology and Evolution_ 10, no. 1 (2019): 70–79. [https://doi.org/10.1111/2041-210X.13112](https://doi.org/10.1111/2041-210X.13112).

Wehner, Helena, Johannes Fritz, and Bernhard Voelkl. ‘Soaring and Intermittent Flap-Gliding during Migratory Flights of Northern Bald Ibis’. _Journal of Ornithology_ (2022). [https://doi.org/10.1007/s10336-022-01978-1](https://doi.org/10.1007/s10336-022-01978-1).