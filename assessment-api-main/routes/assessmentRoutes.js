import express from 'express';
import {createAssessment, getAssessments,updateAssessment} from '../controllers/assessmentFunctions.js';

const routes = express.Router();

routes.route('/').post(createAssessment).get(getAssessments);
routes.route('/:assessmentId').put(updateAssessment)

export default routes;