

const {
  dialogflow, Suggestions,BasicCard,Image,Button
} = require('actions-on-google');

const functions = require('firebase-functions');
const requestNode = require('request');

const app = dialogflow({debug: true});

exports.dialogflowFirebaseFulfillment = functions.https.onRequest(app);


app.intent('Welcome Intent', (conv) => {
  conv.ask(`Hi, We are ScreenLess` );
    conv.close(new BasicCard({
        buttons: new Button({
            title: 'Visit website',
            //add url here
            url: `http://www.pablobiedma.com`,
        }),
        title: 'Pablo Biedma',
        subtitle: 'Passionate Tech Evangelist',
        text: `Here's a link to my website🤖🤘`,
        display: 'CROPPED',
        image: new Image({
            url: 'http://iamakshitgupta.com/wp-content/uploads/2018/08/googleAssistantCoverImage.png',
            alt: 'Personal Image',
        })
    }));
});

app.intent('controlSiteIntent', (conv) => {
    conv.ask(`I love doing amazing things!` );
    conv.ask(`But first, tell me what is the code you see on your screen?`);

});


app.intent('ScrollIntent', (conv,{direction}) => {
    //save command to db here
    if(conv.data.loginStatus) {
        if (direction !== null) {
            if(direction==="up")
            {
            saveToDb("scrollUp");}
            else{
                saveToDb("scrollDown");
            }
            conv.ask("Your request is my command!");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
        }
        else {
            conv.ask("Sorry, but don't be too smart");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
        }
    }
    else{
        conv.ask("Please Login first");
        conv.ask(new Suggestions(['Login']));

    }



  
});

app.intent('navigationIntent',(conv,{element})=>{

    if(conv.data.loginStatus) {
        if (element !== null) {
            if(element==="Scroll Up")
            {
                saveToDb("Scroll Up");
                conv.ask("There you go. What would you like to do next?");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
            }
            else if(element==="Go to Main Dishes"){
                saveToDb("Go to Main Dishes");
                conv.ask("There you go. What would you like to do next?");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
            }
            else if(element==="Scroll Down"){
                saveToDb("Scroll Down");
                conv.ask("There you go. What would you like to do next?");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
            }
            else if(element==="Side Dishes"){
                saveToDb("Side Dishes");
                conv.ask("There you go. What would you like to do next?");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
            }
            else if(element==="Desserts"){
                saveToDb("Desserts");
                conv.ask("There you go. What would you like to do next?");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
            }

        }
        else {
            conv.ask("Please enter a valid command");
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
        }
    }
    else{
        conv.ask("Please Login first");
        conv.ask(new Suggestions(['Login']));

    }


});

app.intent('loginUserIntent', (conv,{number}) => {


    return new Promise(function (resolve, reject) {
        const options = {
            url: 'https://mywebsite-fa9a9.firebaseio.com/command.json',
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },

        };
        requestNode(options, (err, response, body) => {
            if (err) {
                reject(err);
            } else {
                resolve(body);
            }
        });
    }).then((res) => {
        let newRes = JSON.parse(res);
        let userId=newRes.userId;
        if(Number(userId) === Number(number)) {
            conv.ask("Awesome. Now I can control the website with the power of your voice.");
            conv.ask('You can ask me to do things like \'go to side dishes\' or \'scroll down\'. Give it a try!')
            conv.data.loginStatus = true;
            registerLogin();
                conv.ask(new Suggestions(['Scroll Up', 'Scroll Down','Go to Main Dishes','Side Dishes','Desserts']));
        }
        else {
            conv.ask("It doesn't seem to be the right code. Read the code again. Digit after digit. Go!");
        }


    }).catch((err) => {
        conv.close("Sorry, I am having too much traffic right now. Please try after some time.");
    });

});

function saveToDb(command) {
    const options = {
        url: 'https://mywebsite-fa9a9.firebaseio.com/command.json',
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json'
        },
        body : JSON.stringify({
            "command" : command
        })
    };
    requestNode(options, (error, requestInternal, body)=>{
        console.log(body);
 });
}

function registerLogin() {
    const options = {
        url: 'https://mywebsite-fa9a9.firebaseio.com/command.json',
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json'
        },
        body : JSON.stringify({
            "loginStatus" : true
        })
    };
    requestNode(options, (error, requestInternal, body)=>{
        console.log(body);
    });
}