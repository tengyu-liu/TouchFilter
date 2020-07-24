### 07/15/2020
* each time a contact point is predicted, also predict a heatmap on texture map
* when optimizing, use heatmap to weigh loss
* update prediction of heatmap by maximizing heatmap-weighed loss
  * loss includes fc_loss and distance from vertex to target
  * if loss is large, heatmap is penalized more
  * if loss is small, heatmap is penalized less
  * if distance is large, heatmap is infeasible, therefore penalized
  * if distance is small, heatmap is feasible, therefore not penalized
