# 15 Viva Questions & Simple Answers
## Route Optimization System (Smart Navigation for Electric Vehicles)

### Q1: What makes route optimization for Electric Vehicles different from traditional petrol/diesel vehicles?
**Answer:** Traditional cars only care about shortest distance or fastest time. Electric vehicles can recover energy going downhill using **regenerative braking**, consume excessive energy climbing uphill against gravity, and drain battery power for heating or cooling while stuck in traffic. EV route optimization must consider energy consumption (kWh) and battery state.

---

### Q2: Why did you use an Artificial Neural Network (ANN) instead of a simple mathematical formula for energy prediction?
**Answer:** While simple physics equations exist, real-world energy consumption involves complex, non-linear interactions between speed, air resistance, road gradient, payload, traffic stop-and-go behavior, and temperature. The ANN learns these non-linear relationships with high empirical accuracy ($R^2 = 98.59\%$) and fast inference times.

---

### Q3: What is the role of NetworkX in this project?
**Answer:** NetworkX is a Python graph theory library. We use it to model the city as a graph where nodes represent intersections and landmarks, and edges represent roads with attributes like distance, speed limit, traffic level, and slope.

---

### Q4: How does Dijkstra's algorithm work in your project?
**Answer:** Dijkstra's algorithm searches outward from the starting intersection using a min-heap priority queue, always visiting the node with the lowest cumulative cost. Depending on the mode selected, this cost can be distance (km), travel time (minutes), or ANN-predicted energy (kWh).

---

### Q5: What is the difference between Dijkstra's algorithm and A* search?
**Answer:** Dijkstra evaluates all directions equally without knowing where the destination is. A* is an "informed" search that adds an **admissible heuristic** $h(n)$ (estimated straight-line distance to the goal) to the cost $g(n)$:
$$f(n) = g(n) + h(n)$$
This directs the search toward the destination, evaluating fewer nodes while guaranteeing the same optimal path.

---

### Q6: What does it mean for an A* heuristic to be "admissible"?
**Answer:** An admissible heuristic never overestimates the actual cost to reach the goal ($h(n) \le h^*(n)$). Because a straight line is physically the shortest possible distance between two points, Euclidean distance never overestimates road distance, guaranteeing that A* finds the optimal route.

---

### Q7: Why must road graph edge weights be non-negative in your algorithms?
**Answer:** Standard Dijkstra and A* algorithms fail or enter infinite loops if edge weights are negative. Even if an EV recovers energy going downhill, we enforce a tiny positive weight floor ($0.01\text{ kWh}$) to guarantee algorithmic stability.

---

### Q8: What features are fed into your ANN model?
**Answer:** Seven features:
1. `distance_km`: Trip length
2. `speed_kmh`: Average speed
3. `traffic_level_encoded`: Congestion level (0=Low, 1=Medium, 2=High)
4. `slope_percent`: Road incline
5. `elevation_change_m`: Height gained or lost
6. `vehicle_load_kg`: Weight of passengers and cargo
7. `temperature_c`: Outside weather temperature

---

### Q9: What activation functions did you use in the ANN and why?
**Answer:** We used **ReLU** ($\text{ReLU}(z) = \max(0, z)$) in the hidden layers because it enables the network to learn non-linear patterns efficiently without suffering from vanishing gradients, and a **Linear** activation in the output layer because this is a continuous regression task predicting kilowatt-hours.

---

### Q10: How did you prevent data leakage during ANN model training?
**Answer:** We applied `StandardScaler` strictly by fitting it **only on the 80% training set** (`scaler.fit_transform(X_train)`), and then transforming the 20% test set (`scaler.transform(X_test)`). The model never sees test set statistics during training.

---

### Q11: What is the purpose of the "Balanced Route" mode?
**Answer:** Often the most energy-efficient route takes a slow detour, while the fastest route drains too much battery. The Balanced mode combines Energy, Time, and Distance into a single normalized weighted cost function, allowing drivers to balance speed and battery preservation.

---

### Q12: How does the system determine if a route is "Infeasible"?
**Answer:** A route is flagged as `INFEASIBLE` if the predicted energy consumption exceeds the total current energy stored in the battery ($E_{\text{consumed}} > E_{\text{current}}$). The system strictly refuses to claim the car can reach the destination and alerts the driver with the exact shortfall in kWh.

---

### Q13: What is the difference between the ANN and the optional CNN module?
**Answer:** The **CNN** handles computer vision: it processes camera images of roads to classify visual traffic congestion into Low, Medium, or High. The **ANN** handles physical energy modeling: it takes that classified traffic level along with road and vehicle physics to predict battery consumption in kWh.

---

### Q14: How does temperature affect EV energy consumption?
**Answer:** Extreme cold (<18°C) increases battery internal resistance and forces high-power electric heaters (PTC/heat pumps) to run to warm the cabin and battery. Extreme heat (>24°C) requires air conditioning compressors. Both increase auxiliary energy consumption.

---

### Q15: What evaluation metrics did you use to validate your ANN model?
**Answer:** We evaluated the model on an unseen 20% test set (300 trips) using:
- **MAE (Mean Absolute Error)**: $0.1338\text{ kWh}$ (measures average absolute prediction error)
- **MSE (Mean Squared Error)**: $0.0374\text{ kWh²}$ (penalizes large outlier errors)
- **RMSE (Root Mean Squared Error)**: $0.1933\text{ kWh}$
- **R² Score**: $0.9859$ (indicates that 98.59% of variance in energy consumption is explained by the model)
