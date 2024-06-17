import mongoose from 'mongoose';

const jobPostSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true,
    },
    company: {
        type: String,
        required: true,
      },
      location: {
        type: String,
        required: true,
      },
      remote: {
        type: Boolean,
        default: false,
      },
      status: {
      
        type: Boolean,
        default: false,
      },
      postedAt: {
        type: Date,
        default: Date.now,
      },
      applicants: 
      [{
        type:mongoose.Schema.Types.ObjectId,
        ref:"Candidate",
        default: [],
      }], //reffers to the candidate model
      views: {
        type: Number,
        default: 0,
      },
      spent: {
        type: Number,
        default: 0,
      },
      dailyBudget: {
        type: Number,
        default: 0,
      },
      description: {
        type: String,
        required: true,
      },
      screeningQuestions: {
        type: String,
        required: true,
      },
    
    });

    const JobPost = mongoose.model('JobPost', jobPostSchema);
    export default JobPost;
    