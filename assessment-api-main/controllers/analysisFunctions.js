import Evaluation from '../models/analysisModel.js';

const receiveData = async (req, res) => {
    try {
        const evaluation = new Evaluation(req.body);
        await evaluation.save();
        console.log('Data saved successfully:', req.body);
        res.send('Data saved successfully: ' + JSON.stringify(req.body));
    } catch (error) {
        console.error('Error saving data:', error);
        res.status(500).send('Error saving data: ' + error.message);
    }
};

export { receiveData };
