
commands:

java cs362.Learn -mode train -algorithm majority -model_file speech.majority.model -data ./speech/speech.train -task classificationjava cs362.Learn -mode test -model_file speech.majority.model -data ./speech/speech.dev -predictions_file speech.dev.predictions -task classification
java cs362.Learn -mode test -model_file speech.majority.model -data ./speech/speech.test -predictions_file speech.test.predictions -task classificationjava cs362.Learn -mode train -algorithm even_odd -model_file speech.even_odd.model -data ./speech/speech.train -task classificationjava cs362.Learn -mode test -model_file speech.even_odd.model -data ./speech/speech.dev -predictions_file speech.dev.predictions -task classification
java cs362.Learn -mode test -model_file speech.even_odd.model -data ./speech/speech.test -predictions_file speech.test.predictions -task classification