use pyo3::gc::{PyGCProtocol, PyVisit};
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{wrap_pyfunction, PyObjectProtocol, PyTraverseError};
mod pre_pass;
use pre_pass::compute_graph as comp;
use pre_pass::Repetition;
use std::collections::HashMap;
use std::time::Instant;

/// This is the building block of the graph generated by the pre-pass.
/// It contains everything needed to execute the graph.
#[pyclass(gc, module = "pre_pass")]
struct PyDistribution {
	#[pyo3(get, set)]
	dist_type: Option<Py<PyString>>,
	#[pyo3(get, set)]
	ancestors: Option<Py<PyList>>,
	#[pyo3(get, set)]
	rel_influence: f32,
	// Only used in python, set to python's None
	#[pyo3(get, set)]
	mag: Option<PyObject>,
	#[pyo3(get, set)]
	kt_offset_vec: Option<PyObject>,
	// Additional info for debugging, not needed for simulation
	#[pyo3(get)]
	prepass_mag: Option<Py<PyComplex>>,
	#[pyo3(get)]
	influence: f32,
	#[pyo3(get)]
	prepass_signal: f32,
	#[pyo3(get)]
	prepass_rel_signal: f32,
	#[pyo3(get)]
	prepass_kt_vec: Vec<f32>,
}

#[pymethods]
impl PyDistribution {
	#[new]
	fn new<'p>(py: Python<'p>) -> Self {
		PyDistribution {
			dist_type: Some(PyString::new(py, "?").into()),
			ancestors: Some(PyList::empty(py).into()),
			rel_influence: 0.0,
			mag: Some(py.None()),
			kt_offset_vec: Some(py.None()),
			prepass_mag: Some(PyComplex::from_doubles(py, 0.0, 0.0).into()),
			influence: 0.0,
			prepass_signal: 0.0,
			prepass_rel_signal: 0.0,
			prepass_kt_vec: vec![0.0, 0.0, 0.0, 0.0],
		}
	}
}

#[pyproto]
impl PyGCProtocol for PyDistribution {
	fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
		if let Some(dist_type) = &self.dist_type {
			visit.call(dist_type)?;
		}
		if let Some(ancestors) = &self.ancestors {
			visit.call(ancestors)?;
		}
		if let Some(mag) = &self.mag {
			visit.call(mag)?;
		}
		if let Some(kt_offset_vec) = &self.kt_offset_vec {
			visit.call(kt_offset_vec)?;
		}
		if let Some(prepass_mag) = &self.prepass_mag {
			visit.call(prepass_mag)?;
		}
		Ok(())
	}

	fn __clear__(&mut self) {
		self.dist_type = None;
		self.ancestors = None;
		self.mag = None;
		self.kt_offset_vec = None;
		self.prepass_mag = None;
	}
}

#[pyproto]
impl PyObjectProtocol for PyDistribution {
	fn __repr__(&self) -> PyResult<String> {
		// TODO: return more information
		let gil = Python::acquire_gil();
		let py = gil.python();
		Ok(format!(
			"Dist({}, {}, {:?})",
			self.dist_type.as_ref().unwrap().as_ref(py),
			match &self.ancestors {
				Some(a) => a.as_ref(py).len(),
				None => 0,
			},
			self.prepass_kt_vec
		))
	}
}

/// Computes a graph for the given sequence and parameters.
/// seq: Sequence that is simulated
/// mean_t1: Average T1 value for the single-voxel simulation
/// mean_t2: Average T2 value for the single-voxel simulation
/// mean_t2dash: Average T2' value for the single-voxel simulation
/// max_dist_count: Maximum number of + or z distributions simulated
/// min_dist_mag: Minimum absolute magnetisation of a simulated distributions
#[pyfunction]
fn compute_graph<'p>(
	py: Python<'p>,
	seq: &PyList,
	data_shape: [f32; 3],
	mean_t1: f32,
	mean_t2: f32,
	mean_t2dash: f32,
	max_dist_count: usize,
	min_dist_mag: f32,
) -> PyResult<&'p PyList> {
	println!(">>>> Rust - compute_graph(...) >>>");
	let start = Instant::now();
	let mut sequence = Vec::new();

	for rep in seq.iter() {
		let pulse = rep.getattr("pulse")?;
		let pulse_angle: f32 = pulse.getattr("angle")?.extract()?;
		let pulse_phase: f32 = pulse.getattr("phase")?.extract()?;
		let event_count: usize = rep.getattr("event_count")?.extract()?;

		let event_time = rep.getattr("event_time")?.getattr("cpu")?.call0()?;
		let gradm_event = rep.getattr("gradm")?.getattr("cpu")?.call0()?;
		let adc_phase = rep.getattr("adc_phase")?.getattr("cpu")?.call0()?;
		let adc_usage = rep.getattr("adc_usage")?.getattr("cpu")?.call0()?;

		let event_time_addr: usize = event_time.getattr("data_ptr")?.call0()?.extract()?;
		let gradm_addr: usize = gradm_event.getattr("data_ptr")?.call0()?.extract()?;
		let adc_phase_addr: usize = adc_phase.getattr("data_ptr")?.call0()?.extract()?;
		let adc_usage_addr: usize = adc_usage.getattr("data_ptr")?.call0()?.extract()?;

		let event_time =
			unsafe { std::slice::from_raw_parts(event_time_addr as *mut f32, event_count) };
		let gradm = unsafe { std::slice::from_raw_parts(gradm_addr as *mut [f32; 3], event_count) };
		let adc_phase =
			unsafe { std::slice::from_raw_parts(adc_phase_addr as *mut f32, event_count) };
		let adc_usage =
			unsafe { std::slice::from_raw_parts(adc_usage_addr as *mut i32, event_count) };

		let repetition = Repetition {
			pulse_angle,
			pulse_phase,
			event_count: event_count as u32,
			event_time: event_time.to_vec(),
			gradm_event: gradm.to_vec(),
			adc_phase: adc_phase.to_vec(),
			adc_mask: adc_usage.iter().map(|&usage| usage > 0).collect(),
		};

		sequence.push(repetition);
	}

	println!(
		"Converting Python -> Rust: {} s",
		start.elapsed().as_secs_f32()
	);
	let start = Instant::now();
	println!("Computing Graph");

	let mut graph = comp(
		&sequence,
		data_shape,
		mean_t1,
		mean_t2,
		mean_t2dash,
		max_dist_count,
		min_dist_mag,
	);

	println!("Computing Graph: {} s", start.elapsed().as_secs_f32());
	let start = Instant::now();

	// TODO: Rename this, doesn't simplify anymore but calculates stats
	println!("Simplifying Graph");
	pre_pass::simplify_graph(&mut graph);
	println!("Simplyfying Graph: {} s", start.elapsed().as_secs_f32());
	let start = Instant::now();

	let mut mapping: HashMap<usize, Py<PyDistribution>> = HashMap::new();
	let address = |rc: &pre_pass::RcDist| rc.as_ptr() as usize;

	// Create a PyDistribution for every Distribution and store the mapping
	for rep in graph.iter() {
		for dist in rep.iter() {
			// Create the PyDistribution
			let mut py_dist = PyDistribution::new(py);

			// Required for simulation
			py_dist.dist_type = Some(
				PyString::new(
					py,
					pre_pass::DIST_TYPE_STR[dist.borrow().dist_type as usize],
				)
				.into(),
			);
			// Use the mapping to search for ancestors and create python references
			for ancestor in dist.borrow().ancestors.iter() {
				let py_ancestor = mapping.get(&address(&ancestor.dist)).unwrap().clone_ref(py);
				py_dist
					.ancestors
					.as_ref()
					.unwrap()
					.as_ref(py)
					.append(PyTuple::new(
						py,
						&[
							PyString::new(
								py,
								pre_pass::DIST_RELATION_STR[ancestor.relation as usize],
							)
							.as_ref(),
							py_ancestor.as_ref(py),
							PyComplex::from_doubles(
								py,
								ancestor.rot_mat_factor.re as f64,
								ancestor.rot_mat_factor.im as f64,
							)
							.as_ref(),
						],
					))?;
			}
			py_dist.rel_influence = dist.borrow().rel_influence;

			// Extra debugging info
			py_dist.prepass_mag = {
				let mag = dist.borrow().mag;
				Some(PyComplex::from_doubles(py, mag.re as f64, mag.im as f64).into())
			};
			py_dist.influence = dist.borrow().influence;
			py_dist.prepass_signal = dist.borrow().signal;
			py_dist.prepass_rel_signal = dist.borrow().rel_signal;
			py_dist.prepass_kt_vec = dist.borrow().kt_vec.to_vec();
			// Insert the PyDistribution into map
			mapping.insert(address(dist), Py::new(py, py_dist).unwrap());
		}
	}

	let py_graph = PyList::new(
		py,
		graph.into_iter().map(|rep| {
			PyList::new(
				py,
				rep.into_iter()
					.map(|dist| mapping.get(&address(&dist)).unwrap()),
			)
		}),
	);

	println!(
		"Converting Rust -> Python: {} s",
		start.elapsed().as_secs_f32()
	);
	println!("<<<< Rust <<<<");

	Ok(py_graph)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pre_pass(_py: Python, m: &PyModule) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(compute_graph, m)?)?;
	m.add_class::<PyDistribution>()?;

	Ok(())
}
