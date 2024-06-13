import express from 'express';

import {getJobposts,getJobpostsById,createJobPost,updateJobPost,deleteJobPost} from '../controllers/jobPostFunction.js';

const router = express.Router();

router.route('/')
  .get(getJobposts)
  .post(createJobPost);

router.route('/:id')
  .get(getJobpostsById)
  .put(updateJobPost)
  .delete(deleteJobPost);

export default router;
