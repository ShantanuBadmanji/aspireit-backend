import mongoose from 'mongoose';

const candidateSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
    },
    contact_no: {
      type: Number,
      required: false,
      unique: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
    },
    domain: {
      type: String,
      enum: ["Software", "Analyst", "Finance", "Marketing", "HR"],
      default: "Software",
    },
    resume: {
      // pdf
      type: Buffer,
      required: false,
      default: null,
    },
    links: {
      type: [String],
      default: [],
    },
    role: {
      type: String,
      required: true,
    },
    skills: {
      type: String,
      required: true,
    },
    status: {
      type: String,
      required: true,
    },
    active: {
      type: Boolean,
      required: true,
      default: true,
    },
    current: {
      type: String,
      required: true,
    },
    img: {
      type: String,
      required: false,
    },
    percentage: {
      type: Number,
      required: true,
      default: 0,
    },
  
  },
  { timestamps: true }
);

const candidateModel = mongoose.model('Candidate', candidateSchema);

export default candidateModel;
