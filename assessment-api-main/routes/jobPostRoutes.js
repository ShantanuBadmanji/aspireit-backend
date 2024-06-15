import express from 'express';

import {getJobposts,getJobpostsById,createJobPost,updateJobPost,deleteJobPost, applyJobPost} from '../controllers/jobPostFunction.js';

const router = express.Router();

router.route('/')
  .get(getJobposts)
  .post(createJobPost);

router.route('/:id')
  .get(getJobpostsById)
  .put(updateJobPost)
  .delete(deleteJobPost);

router.route('/:id/apply')
  .post(applyJobPost);

export default router;
