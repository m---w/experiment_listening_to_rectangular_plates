
jsPsych.plugins["repeat-audio"] = (function() {

  var plugin = {};

  jsPsych.pluginAPI.registerPreload('repeat-audio', 'stimulus', 'audio');

  plugin.info = {
    name: 'repeat-audio',
    description: '',
    parameters: {
      stimulus: {
        type: [jsPsych.plugins.parameterType.STRING],
        default: undefined,
        no_function: false,
        description: ''
      },
      questions: {
        type: [jsPsych.plugins.parameterType.STRING],
        array: true,
        default: undefined,
        no_function: false,
        description: ''
      },
      options: {
        type: [jsPsych.plugins.parameterType.STRING],
        array: true,
        default: undefined,
        no_function: false,
        description: ''
      },
      correct: {
        type: [jsPsych.plugins.parameterType.STRING],
        array: true,
        default: undefined,
        no_function: false,
        description: ''
      },
      required: {
        type: [jsPsych.plugins.parameterType.BOOL],
        array: true,
        default: false,
        no_function: false,
        description: ''
      },
      force_correct: {
        type: [jsPsych.plugins.parameterType.BOOL],
        array: false,
        default: false,
        no_function: false,
        description: ''
      },
      horizontal: {
        type: [jsPsych.plugins.parameterType.STRING],
        default: 'left',
        no_function: false,
        description: ''
      },
      alignment: {
        type: [jsPsych.plugins.parameterType.BOOL],
        default: false,
        no_function: false,
        description: ''
      },
      preamble: {
        type: [jsPsych.plugins.parameterType.STRING],
        default: '',
        no_function: false,
        description: ''
      },
      superq: {
        type: [jsPsych.plugins.parameterType.STRING],
        default: '',
        no_function: false,
        description: ''
      }

    }
  }

  plugin.trial = function(display_element, trial) {
    // trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);


    var plugin_id_name = "jspsych-survey-multi-choice";
    var plugin_id_selector = '#' + plugin_id_name;
    var _join = function( /*args*/ ) {
      var arr = Array.prototype.slice.call(arguments, _join.length);
      return arr.join(separator = '-');
    }

    // trial defaults
    trial.preamble = typeof trial.preamble == 'undefined' ? "" : trial.preamble;
    trial.superq = typeof trial.superq == 'undefined' ? false : trial.superq;
    trial.required = typeof trial.required == 'undefined' ? null : trial.required;
    trial.force_correct = typeof trial.force_correct == 'undefined' ? true : trial.force_correct;
    trial.horizontal = typeof trial.horizontal == 'undefined' ? false : trial.horizontal;
    trial.alignment = "left";

    // if any trial variables are functions
    // this evaluates the function and replaces
    // it with the output of the function
    // trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);
    var trial_form_id = _join(plugin_id_name, "form");

    var div1 = document.createElement('div');
    display_element.appendChild(div1);
    div1.setAttribute('style', 'text-align:center;')


    var button = document.createElement('button');
    button.setAttribute('type', 'button');
    div1.appendChild(button);
 
    button.setAttribute('style','border-radius:5px;border:0;width:80px;height:80px;background-size:100%;cursor:pointer;background:url(Play.png);background-size:100%')
  


    var p = document.createElement('p');
    p.setAttribute('align', 'center');
    div1.appendChild(p);
    p.innerHTML = 'Replay'

    var div2 = document.createElement('div');
    display_element.appendChild(div2);
    var trial_form = document.createElement('form');
    trial_form.setAttribute('id', trial_form_id);
    div2.appendChild(trial_form);
    

    var node = trial_form.innerHTML += '<style id="jspsych-survey-multi-choice-css">';
    var cssstr = ".jspsych-survey-multi-choice-question { margin-top: 2em; margin-bottom: 2em; text-align: left; }" +
      ".jspsych-survey-multi-choice-text span.required {color: darkred;}" +
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-text {  text-align:" + trial.alignment + ";}" +
      ".jspsych-survey-multi-choice-option { line-height: 2; }" +
      ".jspsych-survey-multi-choice-horizontal .jspsych-survey-multi-choice-option {  display: inline-block;  margin-left: 1em;  margin-right: 1em;  vertical-align: top;}" +
      "label.jspsych-survey-multi-choice-text input[type='radio'] {margin-right: 1em;}"
    trial_form.querySelector('#jspsych-survey-multi-choice-css').innerHTML = cssstr;

    // Audio part:
    var context=jsPsych.pluginAPI.audioContext();
    var source=null;
    var audio =null;
    var click_counter = 0;
    var adjusted_duration = 0.0;

    // load audio file
        jsPsych.pluginAPI.getAudioBuffer(trial.stimulus)
        .then(function (buffer) {
        if (context !== null) {
          console.log('context is NOT null, creating source')
          source = context.createBufferSource();
          source.buffer = buffer;
          source.connect(context.destination);
        } else {
          console.log('context is null, creating audio')
          audio = buffer;
          audio.currentTime = 0;
        }
        //setupTrial();

        if (source !== null) {
          console.log('source is NOT null, adjust duration')
          console.log(source.buffer.duration)
          adjusted_duration = source.buffer.duration * 1000;
          console.log(adjusted_duration)
        }
      
        if (source !== null) {
          console.log('source is NOT null, starting source')
          startTime = context.currentTime;
          source.start(startTime);
        } else {
          console.log('source is null, playing audio')
          audio.play();
        }

     })


    var replayable_audio = function onclick() {
      click_counter +=1
      console.log(click_counter)
      console.log('I was clicked')
      button.setAttribute('disabled', 'true');

      // load audio file
        jsPsych.pluginAPI.getAudioBuffer(trial.stimulus)
        .then(function (buffer) {
        if (context !== null) {
          console.log('context is NOT null, creating source')
          source = context.createBufferSource();
          source.buffer = buffer;
          source.connect(context.destination);
        } else {
          console.log('context is null, creating audio')
          audio = buffer;
          audio.currentTime = 0;
        }
        //setupTrial();

        if (source !== null) {
          console.log('source is NOT null, adjust duration')
          console.log(source.buffer.duration)
          adjusted_duration = source.buffer.duration * 1000;
          console.log(adjusted_duration)
        }
      
        if (source !== null) {
          console.log('source is NOT null, starting source')
          startTime = context.currentTime;
          source.start(startTime);
        } else {
          console.log('source is null, playing audio')
          audio.play();
        }



        })
      
/*
      if (context !== null) {
        console.log('context is NOT null')
        source = context.createBufferSource();
        console.log('source buffer created')
        source.buffer = jsPsych.pluginAPI.getAudioBuffer(trial.stimulus);
        console.log('source buffer done')
        source.connect(context.destination);
        console.log('source connected')

      } else {
        console.log('context is null')
        var audio = jsPsych.pluginAPI.getAudioBuffer(trial.stimulus);
        audio.currentTime = 0;
      }

*/

      

      if (button.disabled) {
        setTimeout(function() {
          button.removeAttribute('disabled')
        }, adjusted_duration)
      }


    }

    button.addEventListener('click', replayable_audio); //this is where the audio stuff ends


    // show preamble text
    var preamble_id_name = _join(plugin_id_name, 'preamble');
    trial_form.innerHTML += '<div id="' + preamble_id_name + '" class="' + preamble_id_name + '">' + trial.preamble + '</div>';

    // show superq text
    if (trial.superq) {
      var superq_id_name = _join(plugin_id_name, 'superq');
      trial_form.innerHTML += '<div id="' + superq_id_name + '" class="' + preamble_id_name + '">' + trial.superq + '</div>';
    }

    // add multiple-choice questions
    for (var i = 0; i < trial.questions.length; i++) {
      // create question container
      var question_classes = [_join(plugin_id_name, 'question')];
      if (trial.horizontal) {
        question_classes.push(_join(plugin_id_name, 'horizontal'));
      }

      trial_form.innerHTML += '<div id="' + _join(plugin_id_name, i) + '" class="' + question_classes.join(' ') + '"></div>';

      var question_selector = _join(plugin_id_selector, i);

      // add question text
      display_element.querySelector(question_selector).innerHTML += '<p id="survey-question" class="' + plugin_id_name + '-text survey-multi-choice">' + trial.questions[i] + '</p>';

      // create option radio buttons
      for (var j = 0; j < trial.options[i].length; j++) {
        var option_id_name = _join(plugin_id_name, "option", i, j),
          option_id_selector = '#' + option_id_name;

        // add radio button container
        display_element.querySelector(question_selector).innerHTML += '<div id="' + option_id_name + '" class="' + _join(plugin_id_name, 'option') + '"></div>';

        // add label and question text
        var form = document.getElementById(option_id_name)
        var input_id_name = _join(plugin_id_name, 'response', i);
        var label = document.createElement('label');
        label.setAttribute('class', plugin_id_name + '-text');
        label.innerHTML = trial.options[i][j];
        label.setAttribute('for', input_id_name)

        // create radio button
        var input = document.createElement('input');
        input.setAttribute('type', "radio");
        input.setAttribute('name', input_id_name);
        input.setAttribute('value', trial.options[i][j])
        label.prepend(input);
        form.appendChild(label)
      }

      if (trial.required && (i < (trial.questions.length) || trial.questions.length == 1)) {
        // if multiple questions, the last one is always the can't tell group, so it shouldn't be required  
        // add required property
        display_element.querySelector(question_selector + " input[type=radio]").required = true;
      }
    }
    // add submit button
    trial_form.innerHTML += '<input type="submit" id="' + plugin_id_name + '-next" class="' + plugin_id_name + ' jspsych-btn" value="Weiter"></input>';

    trial_form.addEventListener('submit', function(event) {
      event.preventDefault();


      var end_trial = function() {
        if (source!==null) {
          source.stop()
        } else if (audio!==null){
          audio.pause();
        }
      };

      end_trial()


      // measure response time     
      var endTime = (new Date()).getTime();
      var response_time = endTime - startTime;

      // create object to hold responses
      var question_data = {};
      var matches = display_element.querySelectorAll("div." + plugin_id_name + "-question");
      for (var i = 0; i < matches.length; i++) {
        match = matches[i];
        var obje = {};
        //  var qid = "Q" + i;
        //    obje[qid] = trial.questions[i];  
        var aid = "A" + i;
        if (match.querySelector("input[type=radio]:checked") !== null) {
          var val = match.querySelector("input[type=radio]:checked").value;
        } else {
          var val = "";
        }
        obje[aid] = val;
        Object.assign(question_data, obje);
      }

      var go_on = function() {
        display_element.innerHTML = ''; // clear the screen
        jsPsych.finishTrial(trial_data);
      }
      //var answer_array = question_data.map(question_data, function(value, index) {
      //  return [value];
      //});

      var answer_array = question_data;

      var iscorrect = JSON.stringify(answer_array) == JSON.stringify(trial.correct)
      if (trial.correct[0] != "NA") {
        // provide feedback
        if (iscorrect) {
          trial_form.innerHTML += '<div>Correct!</div>';
        } else {
          trial_form.innerHTML += '</br>Incorrect. ';
          if (trial.force_correct) {
            trial_form.innerHTML += 'Try again.';
          };
        };
      };
      if (iscorrect || !trial.force_correct || trial.correct[0] == "NA") {
        var trial_data = {
          "replay_count":click_counter,
          "rt": response_time,
          "preamble": trial.preamble,
          "superq": trial.superq,
          "stimulus": trial.stimulus,
          "questions": trial.questions,
          "responses": JSON.stringify(question_data)
        };
        // next trial
        if (trial.correct[0] != "NA") {
          setTimeout(go_on, 1000); // display feedback for 1 sec
        } else {
          go_on() //no feedback, so go right on
        };
      };
    });

    var startTime = (new Date()).getTime();

  };

  return plugin;
})();
