export var getters={
	getLastResult: state => {
		return state.lastResult;
	},
	getModelWeights: state => {
		return state.modelWeights;
	},
	getHistory: state => {
		return state.historyItems;
	},
}