# ARGSPARSE allows you to run python scripts in the command line # 

# Could create a function for it : 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir',type=str,help='GCS location to write checkpoints and export models')
    parser.add_argument('--train-file',type=str,required=True,help='Dataset file local or GCS')
    parser.add_argument('--test-split',type=float,default=0.2,help='Split between training and test, default=0.2')
    parser.add_argument('--num-epochs',type=float,default=500,help='number of times to go through the data, default=500')
    parser.add_argument('--batch-size',type=int,default=128,help='number of records to read during each training step, default=128')
    parser.add_argument('--learning-rate',type=float,default=.001,help='learning rate for gradient descent, default=.001')
    parser.add_argument('--verbosity',choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],default='INFO')
    args, _ = parser.parse_known_args()
    
	return args

    def train_and_evaluate(args):
    (train_data,train_labels), (test_data,test_labels) = load_data(path=args.train_file)
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
    train_steps = args.num_epochs * len(train_data) / args.batch_size
    train_labels = np.asarray(train_labels).astype('float32').reshape((-1, 1))
    train_spec = tf.estimator.TrainSpec(
    	input_fn=lambda: model.input_fn(
    		train_data,
    		train_labels,
    		args.batch_size,
    		mode=tf.estimator.ModeKeys.TRAIN),
    	max_steps=train_steps)
    exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
    test_labels = np.asarray(test_labels).astype('float32').reshape((-1, 1))
    eval_spec = tf.estimator.EvalSpec(
    	input_fn=lambda: model.input_fn(
    		test_data,
    		test_labels,
    		args.batch_size,
    		mode=tf.estimator.ModeKeys.EVAL),
    	steps=None,
    	exporters=[exporter],
    	start_delay_secs=10,
    	throttle_secs=10)
    estimator = model.keras_estimator(
    	model_dir=args.job_dir,
    	config=run_config,
    	params={'learning_rate': args.learning_rate,'num_features': train_data.shape[1]})
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
	# when python interpreter reads a source file will set special variables like __name__ before it then executes code in file. If module being run directly then __name__='main'
	print('argsparse_examples.py is being run directly in the console/wherever')   
	args = get_args()
	tf.logging.set_verbosity(args.verbosity)
	train_and_evaluate(args)
else:
	print('argsparse_examples.py is being imported into another module/script to be run instead. __name__ = nameofyourcurrentmodule instead')
	
	

########### example 2 ##########
def pipeline_orchestration(.......)

def main():
    parser = argparse.ArgumentParser(description='PE scoring')
    parser.add_argument('--setup_file', type=str, help='Path to the json configuration file, to determine storage locations and the specific tuning of the offer allocation')
    
    args = parser.parse_args() 
    
    with open(args.setup_file) as json_file:
        analysis_dicto = json.load(json_file)
        
    pipeline_orchestration(analysis_dicto['SCORING_DATA_PATHS'],
                           analysis_dicto['BUCKET_ID'],
                           analysis_dicto['BUCKET'],
                           analysis_dicto['MODEL_PATH'],
                           analysis_dicto['MODEL_PATH_LOGREG'],
                           analysis_dicto['OPTBIN_PATH'],
                           analysis_dicto['OPTBIN_COLS'],
                           analysis_dicto['MAIN_KEY'],
                           analysis_dicto['PREPRO_OPERATIONS_DICT'],
                           analysis_dicto['ROOT_PATH'],
                           analysis_dicto['DISCOUNT_LEVELS'],
                           analysis_dicto['DISCOUNT_LEVELS_LOGREG'],
                           analysis_dicto['REALLOCATION_PERCENTILES'],
                           analysis_dicto['BATCH_SIZE'],
                           output_filename='PE_UK_DTV_CustomerBase_Scored_July2022_Retrain_5030150',
                           default_csv_filename='PE_UK_scoring_data_tmp_test',
                           log_file=analysis_dicto['LOG_FILE_NAME'])

if __name__ == "__main__":
    main()
