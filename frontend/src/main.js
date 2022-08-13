import Vue from 'vue'
import Vuex from 'vuex'
import App from './App.vue'
import vuetify from './plugins/vuetify'

import firebase from 'firebase/app';
import 'firebase/database';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {tfLib} from "./common/tfLib";

import {config} from './store/config';
import {state} from './store/state';
import {mutations} from './store/mutations';
import {getters} from './store/getters';
import {actions} from './store/actions';

Vue.use(Vuex);
Vue.config.productionTip=false

const fbConfig={
	apiKey: "AIzaSyB2gRl2P10wqom5oQiMUQcPG-SuAtTKtZ8",
	authDomain: "ai-number-recognizer.firebaseapp.com",
	databaseURL: "https://ai-number-recognizer-default-rtdb.firebaseio.com/",
	projectId: "ai-number-recognizer",
	storageBucket: "ai-number-recognizer.appspot.com",
	messagingSenderId: "367947050002",
	appId: "1:367947050002:web:a408b97c90a41d51247ccb",
	measurementId: "G-DGKFFEHXRG"
};

Vue.prototype.$db = firebase.initializeApp(fbConfig).database();
Vue.prototype.$tf = tf;
Vue.prototype.$tfvis = tfvis;

const store=new Vuex.Store({
	state: state,
	getters: getters,
	actions: actions,
	mutations: mutations,
	strict: process.env.NODE_ENV !== 'production' // only by mutations
});

Vue.mixin(tfLib);

new Vue({
	vuetify,
	store,
	render: h => h(App)
}).$mount('#app')
