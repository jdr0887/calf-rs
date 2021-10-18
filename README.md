## Build
```shell
cargo build --release
```

## Run
```shell
RUST_LOG=info ./target/release/calf-rs -i <input>
```

## Example Output
Note the duration at the bottom!
```shell
calf-rs$ RUST_LOG=info ./target/release/calf-rs -i Example_3.csv
[2021-10-18T17:30:49Z INFO  calf_rs] RandomForest - accuracy: 0.6666667, roc_auc_score: 0.6666667, mean_squared_error: 0.33333334
[2021-10-18T17:30:50Z INFO  calf_rs] RandomForest CrossValidation - mean_test_score: 0.5142857, mean_train_score: 0.99846154
[2021-10-18T17:30:50Z INFO  calf_rs] SVC - accuracy: 0.6111111, roc_auc_score: 0.6111111, mean_squared_error: 0.3888889
[2021-10-18T17:30:50Z INFO  calf_rs] SVC CrossValidation - mean_test_score: 0.6214286, mean_train_score: 0.81939906
[2021-10-18T17:30:50Z INFO  calf_rs] LogisticRegression - accuracy: 0.6111111, roc_auc_score: 0.6111111, mean_squared_error: 0.3888889
[2021-10-18T17:30:50Z INFO  calf_rs] LogisticRegression CrossValidation - mean_test_score: 0.65178573, mean_train_score: 0.82867783
[2021-10-18T17:30:50Z INFO  calf_rs] ElasticNet - accuracy: 0, roc_auc_score: 0.60493827, mean_squared_error: 0.23977536
[2021-10-18T17:30:50Z INFO  calf_rs] ElasticNet CrossValidation - mean_test_score: 0.0, mean_train_score: 0.0
[2021-10-18T17:30:50Z INFO  calf_rs] Lasso - accuracy: 0, roc_auc_score: 0.75308645, mean_squared_error: 0.21983545
[2021-10-18T17:30:50Z INFO  calf_rs] Lasso CrossValidation = mean_test_score: 0.0, mean_train_score: 0.0
[2021-10-18T17:30:50Z INFO  calf_rs] KNN - accuracy: 0.6666667, roc_auc_score: 0.6666667, mean_squared_error: 0.33333334
[2021-10-18T17:30:50Z INFO  calf_rs] KNN CrossValidation - mean_test_score: 0.48392853, mean_train_score: 0.74067307
[2021-10-18T17:30:50Z INFO  calf_rs] GaussianNB - accuracy: 0.5555556, roc_auc_score: 0.5555556, mean_squared_error: 0.44444445
[2021-10-18T17:30:50Z INFO  calf_rs] GaussianNB CrossValidation - mean_test_score: 0.63571435, mean_train_score: 0.72524035
[2021-10-18T17:30:50Z INFO  calf_rs] DecisionTree - accuracy: 0.3888889, roc_auc_score: 0.3888889, mean_squared_error: 0.6111111
[2021-10-18T17:30:50Z INFO  calf_rs] DecisionTree CrossValidation - mean_test_score: 0.5500001, mean_train_score: 0.98610574
[2021-10-18T17:30:50Z INFO  calf_rs] Duration: 182ms 662us 957ns
```
