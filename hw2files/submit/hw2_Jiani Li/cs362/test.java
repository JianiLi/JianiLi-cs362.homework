package cs362;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

public class test {

	 public static void main(String args[]){
	        double b [][] = new double[5][5];
	        for(int i = 0; i < b.length; i++) {
	            b[i][i] = 1;  
	        }
	        double y[] = new double[10];
	        for(int i = 0; i < 10; i++) {
	            y[i] = 2;  
	        }
	        RealMatrix matrix_1 = new Array2DRowRealMatrix(y);
	        System.out.println("创建的数组为：\t"+ matrix_1.transpose());
	        //获取矩阵的列数 getColumnDimension() 
	        System.out.println("矩阵的列数为:\t"+ matrix_1.getColumnDimension());
	        //获取矩阵的行数
	        System.out.println("矩阵的行数为:\t"+ matrix_1.getRowDimension());
	        //获取矩阵的某一行,返回,仍然为矩阵
	        RealMatrix matrix_new = new Array2DRowRealMatrix();
	        System.out.println("创建的数组为：\t"+matrix_new);
	        //将数组转化为矩阵
	        RealMatrix matrix = new Array2DRowRealMatrix(b);
	        System.out.println("创建的数组为：\t"+matrix);
	        //获取矩阵的列数 getColumnDimension() 
	        System.out.println("矩阵的列数为:\t"+matrix.getColumnDimension());
	        //获取矩阵的行数
	        System.out.println("矩阵的行数为:\t"+matrix.getRowDimension());
	        //获取矩阵的某一行,返回,仍然为矩阵
	        System.out.println("矩阵的第一行为:\t"+ matrix.getRowMatrix(0));
	        //获取矩阵的某一行,返回,转化为向量
	        System.out.println("矩阵的第一行向量表示为:\t"+ matrix.getRowVector(1) );
	        //矩阵的乘法
	        double testmatrix[][] = new double[2][1];
	        testmatrix[0][0] = 1;
	        testmatrix[1][0] = 2;
	        
	        double testmatrix_another[][] = new double[1][2];
	        testmatrix_another[0][0] = 1;
	        testmatrix_another[0][1] = 2;

	        RealMatrix testmatrix1 = new Array2DRowRealMatrix(testmatrix);
	        RealMatrix testmatrix2 = new Array2DRowRealMatrix(testmatrix_another);
	        System.out.println("两个矩阵相乘后的结果为：\t"+(testmatrix2.multiply(testmatrix1)).getNorm() );
	        //矩阵的转置
	        System.out.println("转置后的矩阵为：\t"+testmatrix1.transpose());
	        //矩阵求逆
	        RealMatrix inversetestMatrix = inverseMatrix(testmatrix1);
	        System.out.println("逆矩阵为：\t"+inversetestMatrix);
	        //矩阵转化为数组 getdata
	        double matrixtoarray[][]=inversetestMatrix.getData();
	        System.out.println("数组中的某一个数字为：\t"+matrixtoarray[0][1]);
	    }
	    //求逆函数
	    public static RealMatrix inverseMatrix(RealMatrix A) {
	        RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
	        return result; 
	    }
	}