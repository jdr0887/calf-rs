#[macro_use]
extern crate log;

use env_logger;
use humantime::format_duration;
use itertools::Itertools;
use plotters::prelude::*;
use smartcore::algorithm::neighbour::KNNAlgorithmName;
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
use smartcore::linear::lasso::{Lasso, LassoParameters};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::math::num::RealNumber;
use smartcore::metrics::{accuracy, mean_squared_error, roc_auc_score};
use smartcore::model_selection::{cross_val_predict, cross_validate, train_test_split, KFold};
use smartcore::naive_bayes::gaussian::{GaussianNB, GaussianNBParameters};
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::Kernels;
use smartcore::tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters};
use std::error;
use std::path;
use std::time;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "calf", about = "calf")]
struct Options {
    #[structopt(short = "i", long = "input", long_help = "input", required = true, parse(from_os_str))]
    input: path::PathBuf,
}
fn main() -> Result<(), Box<dyn error::Error>> {
    let start = time::Instant::now();
    env_logger::init();

    let options = Options::from_args();
    debug!("{:?}", options);

    let (x, y) = load_dataset(&options.input)?;
    scatterplot(&x, None, "x").unwrap();

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25, true);
    scatterplot(&x_train, None, "x_train").unwrap();
    scatterplot(&x_test, None, "x_test").unwrap();

    let rfc_params = RandomForestClassifierParameters::default();
    debug!("RandomForest - params: {:?}", rfc_params);
    let rfc_y_hat = RandomForestClassifier::fit(&x_train, &y_train, rfc_params.clone()).and_then(|rfc| rfc.predict(&x_test)).unwrap();
    info!(
        "RandomForest - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
        accuracy(&y_test, &rfc_y_hat),
        roc_auc_score(&y_test, &rfc_y_hat),
        mean_squared_error(&y_test, &rfc_y_hat)
    );
    let ran_for_cv = cross_validate(RandomForestClassifier::fit, &x, &y, rfc_params.clone(), KFold::default().with_n_splits(10), &accuracy).unwrap();
    info!("RandomForest CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", ran_for_cv.mean_test_score(), ran_for_cv.mean_train_score());
    // let ran_for_cvp = cross_val_predict(RandomForestClassifier::fit, &x, &y, rfc_params.clone(), KFold::default().with_n_splits(10)).unwrap();
    // let xy = DenseMatrix::from_2d_vec(&ran_for_cvp.into_iter().zip(y.clone().into_iter()).map(|(x1, x2)| vec![x1, x2]).collect());
    // scatterplot(&xy, None, "random_forest_cross_val_predict").unwrap();

    let svc_params = SVCParameters::default().with_kernel(Kernels::linear()).with_tol(0.0001);
    debug!("SVC - params: {:?}", svc_params);
    let svc_y_hat = SVC::fit(&x_train, &y_train, svc_params.clone()).and_then(|svc| svc.predict(&x_test)).unwrap();
    info!(
        "SVC - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
        accuracy(&y_test, &svc_y_hat),
        roc_auc_score(&y_test, &svc_y_hat),
        mean_squared_error(&y_test, &svc_y_hat)
    );
    let svc_cv = cross_validate(SVC::fit, &x, &y, svc_params.clone(), KFold::default().with_n_splits(10), &accuracy).unwrap();
    info!("SVC CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", svc_cv.mean_test_score(), svc_cv.mean_train_score());
    // let mesh = make_meshgrid(&x);
    // let asdf = SVC::fit(&x, &y, svc_params.clone()).and_then(|svc| svc.predict(&mesh)).unwrap();
    // scatterplot_with_mesh(&mesh, &asdf, &x, &y, "linear_svm").unwrap();

    let log_reg_params = LogisticRegressionParameters::default();
    debug!("LogisticRegression - params: {:?}", log_reg_params);
    let log_reg_y_hat = LogisticRegression::fit(&x_train, &y_train, log_reg_params.clone()).and_then(|log_reg| log_reg.predict(&x_test)).unwrap();
    info!(
        "LogisticRegression - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
        accuracy(&y_test, &log_reg_y_hat),
        roc_auc_score(&y_test, &log_reg_y_hat),
        mean_squared_error(&y_test, &log_reg_y_hat)
    );
    let log_reg_cv = cross_validate(LogisticRegression::fit, &x, &y, log_reg_params.clone(), KFold::default().with_n_splits(10), &accuracy).unwrap();
    info!("LogisticRegression CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", log_reg_cv.mean_test_score(), log_reg_cv.mean_train_score());

    let elastic_net_params = ElasticNetParameters::default().with_l1_ratio(0.1).with_tol(0.001).with_max_iter(1000).with_normalize(true);
    debug!("LogisticRegression - params: {:?}", log_reg_params);
    let en_y_hat = ElasticNet::fit(&x_train, &y_train, elastic_net_params.clone()).and_then(|enc| enc.predict(&x_test)).unwrap();
    info!(
        "ElasticNet - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
        accuracy(&y_test, &en_y_hat),
        roc_auc_score(&y_test, &en_y_hat),
        mean_squared_error(&y_test, &en_y_hat)
    );
    let log_reg_cv = cross_validate(ElasticNet::fit, &x, &y, elastic_net_params.clone(), KFold::default().with_n_splits(10), &accuracy).unwrap();
    info!("ElasticNet CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", log_reg_cv.mean_test_score(), log_reg_cv.mean_train_score());

    let lasso_params = LassoParameters::default().with_tol(0.01).with_max_iter(1000).with_alpha(0.01).with_normalize(true);
    debug!("Lasso - params: {:?}", lasso_params);
    match Lasso::fit(&x_train, &y_train, lasso_params.clone()) {
        Ok(lasso_classifier) => {
            let lasso_y_hat = lasso_classifier.predict(&x_test).unwrap();
            info!(
                "Lasso - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
                accuracy(&y_test, &lasso_y_hat),
                roc_auc_score(&y_test, &lasso_y_hat),
                mean_squared_error(&y_test, &lasso_y_hat)
            );
            let lasso_cv_results = cross_validate(Lasso::fit, &x, &y, lasso_params.clone(), KFold::default().with_n_splits(10), &accuracy)?;
            info!("Lasso CrossValidation = mean_test_score: {:?}, mean_train_score: {:?}", lasso_cv_results.mean_test_score(), lasso_cv_results.mean_train_score());
        }
        Err(err) => {
            warn!("failed to run Lasso: {}", err.to_string())
        }
    }

    let knn_params = KNNClassifierParameters::default().with_algorithm(KNNAlgorithmName::CoverTree).with_k(3);
    debug!("KNN - params: {:?}", knn_params);
    let knn_y_hat = KNNClassifier::fit(&x_train, &y_train, knn_params.clone()).and_then(|knnc| knnc.predict(&x_test)).unwrap();
    info!(
        "KNN - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
        accuracy(&y_test, &knn_y_hat),
        roc_auc_score(&y_test, &knn_y_hat),
        mean_squared_error(&y_test, &knn_y_hat)
    );
    let knn_cv_results = cross_validate(KNNClassifier::fit, &x, &y, knn_params.clone(), KFold::default().with_n_splits(10), &accuracy)?;
    info!("KNN CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", knn_cv_results.mean_test_score(), knn_cv_results.mean_train_score());

    let nb_params = GaussianNBParameters::default();
    debug!("nb_params: {:?}", nb_params);
    match GaussianNB::fit(&x_train, &y_train, nb_params.clone()) {
        Ok(nb_classifier) => {
            let nb_y_hat = nb_classifier.predict(&x_test).unwrap();
            info!(
                "GaussianNB - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
                accuracy(&y_test, &nb_y_hat),
                roc_auc_score(&y_test, &nb_y_hat),
                mean_squared_error(&y_test, &nb_y_hat)
            );
            let nb_cv_results = cross_validate(GaussianNB::fit, &x, &y, nb_params.clone(), KFold::default().with_n_splits(10), &accuracy)?;
            info!("GaussianNB CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", nb_cv_results.mean_test_score(), nb_cv_results.mean_train_score());
        }
        Err(err) => {
            warn!("failed to run GaussianNB: {}", err.to_string())
        }
    }

    let dt_params = DecisionTreeClassifierParameters::default();
    debug!("DecisionTree - params: {:?}", dt_params.clone());
    let dt_y_hat = DecisionTreeClassifier::fit(&x_train, &y_train, dt_params.clone()).and_then(|dtc| dtc.predict(&x_test)).unwrap();
    info!(
        "DecisionTree - accuracy: {}, roc_auc_score: {}, mean_squared_error: {}",
        accuracy(&y_test, &dt_y_hat),
        roc_auc_score(&y_test, &dt_y_hat),
        mean_squared_error(&y_test, &dt_y_hat)
    );
    let dt_cv_results = cross_validate(DecisionTreeClassifier::fit, &x, &y, dt_params.clone(), KFold::default().with_n_splits(10), &accuracy)?;
    info!("DecisionTree CrossValidation - mean_test_score: {:?}, mean_train_score: {:?}", dt_cv_results.mean_test_score(), dt_cv_results.mean_train_score());

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}

pub fn load_dataset(input: &path::PathBuf) -> Result<(DenseMatrix<f32>, Vec<f32>), Box<dyn error::Error>> {
    let mut rdr = csv::Reader::from_path(input)?;

    let mut data: Vec<Vec<f32>> = Vec::new();
    let mut target: Vec<f32> = Vec::new();

    // let feature_names = rdr.headers()?.iter().skip(1).map(|a| a.to_string()).collect_vec();
    // let target_names = vec![rdr.headers()?.iter().nth(0).take().unwrap().to_string()];

    for result in rdr.records() {
        let record = result?;
        data.push(
            record
                .iter()
                .skip(1)
                .map(|a| match a.parse::<f32>() {
                    Ok(value) => value,
                    Err(_) => 0_f32,
                })
                .collect_vec(),
        );
        target.push(record.iter().nth(0).map(|a| a.parse::<f32>().unwrap()).take().unwrap());
    }

    Ok((DenseMatrix::from_2d_vec(&data), target))
}

/// Get min value of `x` along axis `axis`
pub fn min<T: RealNumber>(x: &DenseMatrix<T>, axis: usize) -> T {
    let n = x.shape().0;
    x.slice(0..n, axis..axis + 1).iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

/// Get max value of `x` along axis `axis`
pub fn max<T: RealNumber>(x: &DenseMatrix<T>, axis: usize) -> T {
    let n = x.shape().0;
    x.slice(0..n, axis..axis + 1).iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
}

/// Draw a mesh grid defined by `mesh` with a scatterplot of `data` on top
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
pub fn scatterplot_with_mesh(mesh: &DenseMatrix<f32>, mesh_labels: &Vec<f32>, data: &DenseMatrix<f32>, labels: &Vec<f32>, title: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;
    let root = root.margin(15, 15, 15, 15);

    let x_min = (min(mesh, 0) - 1.0) as f64;
    let x_max = (max(mesh, 0) + 1.0) as f64;
    let y_min = (min(mesh, 1) - 1.0) as f64;
    let y_max = (max(mesh, 1) + 1.0) as f64;

    let mesh_labels: Vec<usize> = mesh_labels.into_iter().map(|&v| v as usize).collect();
    let mesh: Vec<f64> = mesh.iter().map(|v| v as f64).collect();

    let labels: Vec<usize> = labels.into_iter().map(|&v| v as usize).collect();
    let data: Vec<f64> = data.iter().map(|v| v as f64).collect();

    let mut scatter_ctx = ChartBuilder::on(&root).x_label_area_size(20).y_label_area_size(20).build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    scatter_ctx.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;
    scatter_ctx.draw_series(
        mesh.chunks(2).zip(mesh_labels.iter()).map(|(xy, &l)| EmptyElement::at((xy[0], xy[1])) + Circle::new((0, 0), 1, ShapeStyle::from(&Palette99::pick(l)).filled())),
    )?;
    scatter_ctx.draw_series(
        data.chunks(2).zip(labels.iter()).map(|(xy, &l)| EmptyElement::at((xy[0], xy[1])) + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(l + 3)).filled())),
    )?;

    Ok(())
}

/// Draw a scatterplot of `data` with labels `labels`
/// We use Plotters library to draw scatter plot.
/// https://docs.rs/plotters/0.3.0/plotters/
pub fn scatterplot(data: &DenseMatrix<f32>, labels: Option<&Vec<usize>>, title: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}.svg", title);
    let root = SVGBackend::new(&path, (800, 600)).into_drawing_area();

    let x_min = (min(data, 0) - 1.0) as f64;
    let x_max = (max(data, 0) + 1.0) as f64;
    let y_min = (min(data, 1) - 1.0) as f64;
    let y_max = (max(data, 1) + 1.0) as f64;

    root.fill(&WHITE)?;
    let root = root.margin(15, 15, 15, 15);

    let data_values: Vec<f64> = data.iter().map(|v| v as f64).collect();

    let mut scatter_ctx = ChartBuilder::on(&root).x_label_area_size(20).y_label_area_size(20).build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    scatter_ctx.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;
    match labels {
        Some(labels) => {
            scatter_ctx.draw_series(data_values.chunks(2).zip(labels.iter()).map(|(xy, &l)| {
                EmptyElement::at((xy[0], xy[1]))
                    + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(l)).filled())
                    + Text::new(format!("{}", l), (6, 0), ("sans-serif", 15.0).into_font())
            }))?;
        }
        None => {
            scatter_ctx.draw_series(data_values.chunks(2).map(|xy| EmptyElement::at((xy[0], xy[1])) + Circle::new((0, 0), 3, ShapeStyle::from(&Palette99::pick(3)).filled())))?;
        }
    }

    Ok(())
}

/// Generates 2x2 mesh grid from `x`
pub fn make_meshgrid(x: &DenseMatrix<f32>) -> DenseMatrix<f32> {
    let n = x.shape().0;
    let x_min = min(x, 0) - 1.0;
    let x_max = max(x, 0) + 1.0;
    let y_min = min(x, 1) - 1.0;
    let y_max = max(x, 1) + 1.0;

    let x_step = (x_max - x_min) / n as f32;
    let x_axis: Vec<f32> = (0..n).map(|v| (v as f32 * x_step) + x_min).collect();
    let y_step = (y_max - y_min) / n as f32;
    let y_axis: Vec<f32> = (0..n).map(|v| (v as f32 * y_step) + y_min).collect();

    let x_new: Vec<Vec<f32>> = x_axis.clone().into_iter().flat_map(move |v1| y_axis.clone().into_iter().map(move |v2| vec![v1, v2])).collect();

    DenseMatrix::from_2d_vec(&x_new)
}
